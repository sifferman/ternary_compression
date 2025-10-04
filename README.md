# Ternary Compression LUT Generator

This repo uses monte carlo runs to attempt to generate an optimal LUT to decompress 5 ternary weights out of 8-bits.

If you compress on 8 bit boundaries, instead of just storing 4 weights in 8 bits, we could store 5 weights in 8 bits. In other words, instead of storing each weight as follows:

$$
W_{0}4^{0}+W_{1}4^{1}+W_{2}4^{2}+W_{3}4^{3}
$$

We can instead store each weight as

$$
W_{0}3^{0}+W_{1}3^{1}+W_{2}3^{2}+W_{3}3^{3}+W_{4}3^{4}
$$

However, decompressing is non-trivial, and there are ${256 \choose 13} \approx 2.4 \times 10^{21}$ possible LUTs. Some LUTs use fewer resources than others, so this repo attempts to find an optimal one.

```bash
python3 ./mc.py --jobs `nproc` \
                --runs 10000 \
                --low-heat 0.06 \
                --low-heat-iterations 50 \
                --high-heat 0.4 \
                --high-heat-iterations 2 \
                --num-heat-cycles 2
```
