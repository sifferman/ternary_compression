#!/usr/bin/env python3
"""
Monte Carlo hole optimizer
https://chatgpt.com/share/68e14257-6a60-8007-84e7-e3f390e1edc7
"""

import os
import json
import random
import threading
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# build LUT_VALUES: numbers 0..1023 but skip any that have digit '2' in base-4 (length 243)
LUT_VALUES = []
for n in range(0, 1024):
    x = n
    has_two = False
    for _ in range(5):
        if x % 4 == 2:
            has_two = True
            break
        x //= 4
    if not has_two:
        LUT_VALUES.append(n)
assert len(LUT_VALUES) == 3 ** 5 == 243

# Global best (protected by lock)
best_holes = [i for i in range(13)]
best_holes_lock = threading.Lock()
best_holes_filename = "best_holes.txt"
global_best_score = None

# memh directory
MEMH_DIR = "memh_files"
os.makedirs(MEMH_DIR, exist_ok=True)


def _basename_for_holes(holes):
    return "_".join(map(str, holes))


def _memh_path_for_holes(holes):
    return os.path.join(MEMH_DIR, _basename_for_holes(holes) + ".memh")


def _stat_path_for_holes(holes):
    return os.path.join(MEMH_DIR, _basename_for_holes(holes) + ".stat.json")


def get_utilization(holes):
    """
    Read memh_files/<basename>.stat.json modules->"\lut"->num_cells if present.
    Otherwise write memh_files/<basename>.memh (with 'xxx' in hole slots), run yosys,
    and then read the stat file. Returns int or None.
    """
    if not holes or len(holes) != 13:
        raise ValueError("holes must be a list of 13 integers")

    stat_path = _stat_path_for_holes(holes)

    # try to read existing stat json
    if os.path.exists(stat_path):
        try:
            with open(stat_path, "r") as fh:
                data = json.load(fh)
            num_cells = data.get("modules", {}).get("\\lut", {}).get("num_cells")
            if isinstance(num_cells, int):
                return num_cells
        except Exception:
            pass  # fall through to regenerate

    # build full LUT: insert 'xxx' at hole indices
    full = []
    lut_iter = iter(LUT_VALUES)
    hole_set = set(holes)
    for i in range(256):
        if i in hole_set:
            full.append("xxx")
        else:
            full.append(next(lut_iter))
    assert len(full) == 256

    # write memh file
    memh_path = _memh_path_for_holes(holes)
    with open(memh_path, "w") as fh:
        for val in full:
            if val == "xxx":
                fh.write("xxx\n")
            else:
                fh.write("{:03x}\n".format(int(val)))

    # yosys commands (write stat json into memh_files)
    yosys_cmds = (
        f'read_verilog -DMEMH_FILENAME="{memh_path}" lut.v; '
        "synth; "
        "opt -full; "
        "aigmap; "
        "opt -full; "
        f'tee -o "{stat_path}" stat -json'
    )

    try:
        subprocess.run(
            ["yosys", "-p", yosys_cmds],
            text=True,
            capture_output=True,
            timeout=60,
        )
    except Exception:
        return None

    if os.path.exists(stat_path):
        try:
            with open(stat_path, "r") as fh:
                data = json.load(fh)
            num_cells = data.get("modules", {}).get("\\lut", {}).get("num_cells")
            if isinstance(num_cells, int):
                return num_cells
        except Exception:
            return None

    return None


def change_hole(holes, index):
    """
    Return a new holes list copy where only holes[index] is changed.
    Move it to a new random location strictly between neighbors.
    For edges, left is -1 and right is 256.
    Ensure a change when possible.
    """
    if not (0 <= index < len(holes)):
        raise IndexError("index out of range for holes")

    old = holes[index]
    left = holes[index - 1] if index > 0 else -1
    right = holes[index + 1] if index < len(holes) - 1 else 256

    possible = list(range(left + 1, right))
    if not possible:
        return holes.copy()

    if len(possible) == 1:
        new_val = possible[0]
    else:
        new_val = old
        tries = 0
        while new_val == old and tries < 20:
            new_val = random.choice(possible)
            tries += 1

    new_holes = holes.copy()
    new_holes[index] = new_val
    return sorted(new_holes)


def _log_new_global_best(holes, score):
    """Append a line to best_holes_filename. Called while holding best_holes_lock."""
    with open(best_holes_filename, "a") as fh:
        fh.write(f"score: {score} holes: {'_'.join(map(str, holes))}\n")
        fh.flush()
        os.fsync(fh.fileno())


def _maybe_update_global_best(candidate, cand_score):
    """Update global best (and log) if candidate improves it. Must be called under lock."""
    global best_holes, global_best_score
    if global_best_score is None or cand_score < global_best_score:
        best_holes = candidate.copy()
        global_best_score = cand_score
        _log_new_global_best(best_holes, global_best_score)
        print(f"New global best: score={global_best_score} holes={best_holes}", flush=True)


def _run_until_frozen(holes, current_score, local_best_score, local_best_holes, heat, patience):
    """
    Run proposals until no_improve reaches patience. Progress is measured per full pass:
    - For each pass: attempt to change every hole in order.
    - During the pass, updates to local_best_score/local_best_holes/current_score/holes happen as usual.
    - After the full pass, if local_best_score improved compared to before the pass,
      reset no_improve = 0; otherwise increment no_improve by 1.
    Prints current score & holes after every pass.
    Returns updated (holes, current_score, local_best_score, local_best_holes).
    """
    no_improve = 0
    pass_no = 0
    while no_improve < patience:
        prev_local_best = local_best_score
        pass_no += 1

        # do one full pass: test every hole
        for idx in range(len(holes)):
            candidate = change_hole(holes, idx)
            cand_score = get_utilization(candidate)
            if cand_score is None:
                cand_score = 10 ** 12

            # If candidate is a new local best or equal, always take it
            if cand_score <= local_best_score:
                local_best_score = cand_score
                local_best_holes = candidate.copy()
                holes = candidate
                current_score = cand_score
                # possibly update global best (under lock)
                with best_holes_lock:
                    _maybe_update_global_best(candidate, cand_score)
                continue

            # acceptance rules relative to current_score:
            if cand_score < current_score:
                # ALWAYS accept better (ignore heat)
                holes = candidate
                current_score = cand_score
            elif cand_score > current_score:
                # worse: accept with probability heat
                if random.random() < heat:
                    holes = candidate
                    current_score = cand_score
            # equal to current_score: do nothing

        # end of full pass: decide whether to reset no_improve
        if local_best_score < prev_local_best:
            no_improve = 0
        else:
            no_improve += 1

        # debug: print status after each full pass
        # print(
        #     f"[low-run] pass={pass_no} heat={heat} current_score={current_score} holes={holes} local_best_score={local_best_score}",
        #     flush=True,
        # )

    return holes, current_score, local_best_score, local_best_holes


def _run_fixed_iterations(holes, current_score, local_best_score, local_best_holes, heat, iterations):
    """
    Run a fixed number of passes (iterations) over all hole indices using `heat`.
    - For each pass: loop over indices and propose.
    - Update local best if found (and keep it).
    Prints current score & holes after every pass.
    Returns updated (holes, current_score, local_best_score, local_best_holes).
    """
    for pass_no in range(1, iterations + 1):
        for idx in range(len(holes)):
            candidate = change_hole(holes, idx)
            cand_score = get_utilization(candidate)
            if cand_score is None:
                cand_score = 10 ** 12

            # If candidate <= local_best_score, always take and update local best
            if cand_score <= local_best_score:
                local_best_score = cand_score
                local_best_holes = candidate.copy()
                holes = candidate
                current_score = cand_score
                with best_holes_lock:
                    _maybe_update_global_best(candidate, cand_score)
                continue

            if cand_score < current_score:
                # always accept better
                holes = candidate
                current_score = cand_score
            elif cand_score > current_score:
                # accept worse with probability heat
                if random.random() < heat:
                    holes = candidate
                    current_score = cand_score
            # equal -> do nothing

        # debug: print status after each full pass
        # print(
        #     f"[high-run] pass={pass_no}/{iterations} heat={heat} current_score={current_score} holes={holes} local_best_score={local_best_score}",
        #     flush=True,
        # )
    return holes, current_score, local_best_score, local_best_holes


def monte_carlo(
    holes=None,
    low_heat=0.1,
    low_heat_iterations=10,
    high_heat=0.5,
    high_heat_iterations=10,
    num_heat_cycles=1,
):
    """
    Monte Carlo with alternating low/high heat cycles.

    Returns {"holes": local_best_holes, "score": local_best_score} and prints local best on exit.
    """
    global best_holes, global_best_score

    if holes is None:
        holes = sorted(random.sample(range(256), 13))
    else:
        holes = sorted(holes)

    # initial evaluation
    current_score = get_utilization(holes)
    if current_score is None:
        current_score = 10 ** 12

    local_best_score = current_score
    local_best_holes = holes.copy()

    # print(f"Starting monte_carlo: initial_score={current_score} holes={holes}", flush=True)

    # print(f"Entering low-heat phase: heat={low_heat}, patience={low_heat_iterations}", flush=True)
    holes, current_score, local_best_score, local_best_holes = _run_until_frozen(
        holes, current_score, local_best_score, local_best_holes, low_heat, low_heat_iterations
    )

    for cycle in range(1, num_heat_cycles + 1):
        # print(f"Cycle {cycle}/{num_heat_cycles}: switching to high heat={high_heat} for {high_heat_iterations} passes", flush=True)
        # high heat: fixed iterations
        holes, current_score, local_best_score, local_best_holes = _run_fixed_iterations(
            holes, current_score, local_best_score, local_best_holes, high_heat, high_heat_iterations
        )

        # print(f"Cycle {cycle}/{num_heat_cycles}: switching back to low heat={low_heat} until frozen", flush=True)
        # low heat until frozen
        holes, current_score, local_best_score, local_best_holes = _run_until_frozen(
            holes, current_score, local_best_score, local_best_holes, low_heat, low_heat_iterations
        )

    print(f"Finished monte_carlo: local_best_score={local_best_score} local_best_holes={local_best_holes}", flush=True)
    return {"holes": local_best_holes, "score": local_best_score}


def main(
    jobs=4,
    runs=8,
    low_heat=0.1,
    low_heat_iterations=10,
    high_heat=0.5,
    high_heat_iterations=10,
    num_heat_cycles=1,
):
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [
            ex.submit(
                monte_carlo,
                None,
                low_heat,
                low_heat_iterations,
                high_heat,
                high_heat_iterations,
                num_heat_cycles,
            )
            for _ in range(runs)
        ]
        for f in as_completed(futures):
            try:
                res = f.result()
            except Exception as e:
                print("Task failed:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo hole optimizer with heat cycles")
    parser.add_argument("--jobs", type=int, default=4, help="number of concurrent threads")
    parser.add_argument("--runs", type=int, default=8, help="total monte carlo runs to execute")
    parser.add_argument("--low-heat", type=float, default=0.1, help="low heat (accept worse with this probability)")
    parser.add_argument(
        "--low-heat-iterations",
        type=int,
        default=10,
        help="patience for low heat: number of non-improving full passes before quitting low-heat phase",
    )
    parser.add_argument("--high-heat", type=float, default=0.5, help="high heat (accept worse with this probability)")
    parser.add_argument(
        "--high-heat-iterations",
        type=int,
        default=10,
        help="number of full-index passes to run during high-heat phase",
    )
    parser.add_argument(
        "--num-heat-cycles",
        type=int,
        default=1,
        help="number of high-heat -> low-heat cycles to perform after the initial low-heat freeze",
    )
    parser.add_argument("--best-file", type=str, default="best_holes.txt", help="file to store best holes")
    args = parser.parse_args()

    best_holes_filename = args.best_file
    main(
        jobs=args.jobs,
        runs=args.runs,
        low_heat=args.low_heat,
        low_heat_iterations=args.low_heat_iterations,
        high_heat=args.high_heat,
        high_heat_iterations=args.high_heat_iterations,
        num_heat_cycles=args.num_heat_cycles,
    )
