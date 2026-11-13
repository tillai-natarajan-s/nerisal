#!/usr/bin/env python3
"""
Build proof.json (the dashboard's Results drawer) from our own measured results.

  python make_proof.py --bench bench_india_summary.csv bench_hybrid_summary.csv \
                       --set "18 hand-counted Indian photos" --speed speed_pi5.csv --out proof.json

Only real numbers from benchmark.py and speed_ladder.py go in. Nothing is typed by hand.
"""
import argparse, csv, json, os, platform, re


def nice(method):
    m = method.replace(".pte", "")
    m = re.sub(r"^yolo (\S+) @\d+ single$", r"\1, single pass", m)
    m = re.sub(r"^yolo (\S+) @\d+ tiled (\dx\d)$", r"\1, tiled \2", m)
    m = re.sub(r"^dmcount (\S+) tiles (\dx\d)$", r"DM-Count, \2 tiles", m)
    m = re.sub(r"^hybrid (\S+) \+ (\S+)$", r"Hybrid: \1 near, DM-Count far", m)
    return m


def device():
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n")
    except OSError:
        return platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=[], help="benchmark.py *_summary.csv files")
    ap.add_argument("--set", default="hand-counted photos", help="name of the test image set")
    ap.add_argument("--speed", default="", help="speed_ladder.py --out CSV")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default="proof.json")
    a = ap.parse_args()

    acc, seen, extra_speed = [], set(), []
    for path in a.bench:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["band"] != "ALL" or r["method"] in seen:
                    continue
                seen.add(r["method"])
                acc.append({"method": nice(r["method"]), "set": a.set, "n": int(r["n"]),
                            "mae": float(r["MAE"]), "pct": float(r["mean_pct_err"]),
                            "median_pct": float(r.get("median_pct_err") or 0), "bias": float(r["bias"])})
                if "avg_ms" in r and ("tiled" in r["method"] or "hybrid" in r["method"]):
                    extra_speed.append({"model": nice(r["method"]), "input": "full pipeline, per image",
                                        "ms": float(r["avg_ms"]), "fps": round(1000 / float(r["avg_ms"]), 2)})
    spd = []
    if a.speed and os.path.exists(a.speed):
        with open(a.speed) as f:
            for r in csv.DictReader(f):
                spd.append({"model": r["model"].replace(".pte", ""), "input": r["input"].replace("x", " × "),
                            "ms": float(r["mean_ms"]), "fps": float(r["fps"])})
    proof = {"device": device(), "accuracy": acc, "speed": spd + extra_speed,
             "notes": a.notes or ("Average error is the mean absolute difference from the true count. "
                                  "All numbers measured by tools/benchmark.py and tools/speed_ladder.py.")}
    with open(a.out, "w") as f:
        json.dump(proof, f, indent=1)
    print(f"saved {a.out}: {len(acc)} accuracy rows, {len(proof['speed'])} speed rows")


if __name__ == "__main__":
    main()
