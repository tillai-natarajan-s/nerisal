# Results

Every number in this repository was measured on a Raspberry Pi 5 with the scripts in `tools/`. Nothing is estimated.

| File | What it is | Made by |
|---|---|---|
| `hand_counts_india.csv` | True head counts for 18 Indian street photos (every head marked by hand) | Team Iconic |
| `hand_counts_india_sources.txt` | Where each photo comes from (Wikimedia Commons, Pexels, Unsplash). Photos are not redistributed here. | |
| `benchmark/bench_india_*` | Single, tiled and density methods on the hand-counted set | `tools/benchmark.py` |
| `benchmark/bench_hybrid_*` | Hybrid counter against tiled detection | `tools/benchmark.py` |
| `benchmark/bench_int8_*` | Hybrid with FP32 against INT8 head models | `tools/benchmark.py` |
| `speed/speed_pi5.csv` | Every model timed on the Pi 5 | `tools/speed_ladder.py` |
| `training/` | Training curves, confusion matrix, count check and label check for head640 and head320 | notebook 01 |
| `proof.json` | The dashboard's Results drawer, built from the files above | `tools/make_proof.py` |

`*_per_image.csv` files list every photo with the true count and each method's count, so any number in the
main README can be traced back to individual images.
