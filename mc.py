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

# build LUT_VALUES: 0..1023 but skip any that have digit '2' in base-4 (length 243)
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
    Try to read memh_files/<basename>.stat.json and return design.num_cells from
    modules -> "\lut" -> num_cells. If not present, write the .memh (with 'xxx'
    in hole slots), run yosys (commands via -p), then read the stat json.
    Returns int or None on failure.
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
    Move it to a new random location strictly between neighbors (left+1 .. right-1).
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


def monte_carlo(holes=None, heat=0.1, iterations=10):
    """
    Monte Carlo with local-best/patience stopping:
    - holes: initial holes or randomize if None
    - heat: acceptance probability for worse proposals
    - iterations: patience threshold (number of non-improving evaluations before quit)

    The function:
      - tracks local_best_score and local_best_holes
      - resets the no-improve counter to 0 whenever a candidate meets or beats local_best_score
      - quits only if it hasn't found a new local_best_score in `iterations` evaluations
      - prints its local best on exit and returns it
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

    no_improve = 0  # counts how many evaluations since last local best
    # continue until no_improve reaches iterations
    while no_improve < iterations:
        for idx in range(len(holes)):
            if no_improve >= iterations:
                break

            candidate = change_hole(holes, idx)
            cand_score = get_utilization(candidate)
            if cand_score is None:
                cand_score = 10 ** 12

            # If candidate meets or beats the local best, always take it and reset counter
            if cand_score <= local_best_score:
                local_best_score = cand_score
                local_best_holes = candidate.copy()
                holes = candidate
                current_score = cand_score
                no_improve = 0

                # update global best if surpassed (under lock) — this is the only place we read global_best_score
                with best_holes_lock:
                    if global_best_score is None or cand_score < global_best_score:
                        best_holes = candidate.copy()
                        global_best_score = cand_score
                        _log_new_global_best(best_holes, global_best_score)
                        print(f"New global best: score={global_best_score} holes={best_holes}", flush=True)
                continue

            # otherwise, acceptance based on comparison with current_score
            if cand_score < current_score:
                # better than current: accept with prob (1 - heat)
                if random.random() < (1.0 - heat):
                    holes = candidate
                    current_score = cand_score
            elif cand_score > current_score:
                # worse than current: accept with prob heat
                if random.random() < heat:
                    holes = candidate
                    current_score = cand_score
            # if equal to current_score but not <= local_best_score, do nothing

            no_improve += 1

    # finished: print and return local best
    print(f"Finished monte_carlo: local_best_score={local_best_score} local_best_holes={local_best_holes}", flush=True)
    return {"holes": local_best_holes, "score": local_best_score}


def main(jobs=4, runs=8, heat=0.1, iterations=10):
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(monte_carlo, None, heat, iterations) for _ in range(runs)]
        for f in as_completed(futures):
            try:
                res = f.result()
            except Exception as e:
                print("Task failed:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo hole optimizer")
    parser.add_argument("--jobs", type=int, default=4, help="number of concurrent threads")
    parser.add_argument("--runs", type=int, default=8, help="total monte carlo runs to execute")
    parser.add_argument("--heat", type=float, default=0.1, help="heat parameter (0..1)")
    parser.add_argument("--iterations", type=int, default=10, help="patience: number of non-improving evaluations before quitting")
    parser.add_argument("--best-file", type=str, default="best_holes.txt", help="file to store best holes")
    args = parser.parse_args()

    best_holes_filename = args.best_file
    main(jobs=args.jobs, runs=args.runs, heat=args.heat, iterations=args.iterations)
