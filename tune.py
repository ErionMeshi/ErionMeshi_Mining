"""
tune.py - kerkon automatikisht parametra me te mire per beam.py (random search).

Vendose ne te njejtin folder me beam.py dhe udp.py. Perdorimi:
    python tune.py --trials 20 --starts 3 --beam 30
Ruan solution_tuned.json sa here gjen rezultat me te mire (objektiv me negativ = me mire).
"""
import argparse
import json
import time
import numpy as np
import beam as B
import udp as U


def evaluate(starts, width, rng):
    best_x, best_f = None, 1e9
    for _ in range(starts):
        start = int(rng.integers(B.N))
        s = B.beam_search(start, width, verbose=False)
        x = B.to_x(s)
        f = U.udp.fitness(x)
        ok = f[1] == 0 and max(f[2:]) <= 0
        if ok and f[0] < best_f:
            best_x, best_f = x, f[0]
    return best_x, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--beam", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    overall_x, overall_f = None, 1e9

    for t in range(args.trials):
        B.LT = float(rng.choice([0.003, 0.0045, 0.0055, 0.007, 0.01]))
        B.LF = float(rng.choice([0.2, 0.5, 1.0, 1.5]))
        B.RESERVE = float(rng.choice([0.05, 0.1, 0.15, 0.25]))
        B.K_NEAR = int(rng.choice([30, 40, 60]))
        B.K_PER_MAT = int(rng.choice([6, 8, 12]))
        t0 = time.time()
        x, f = evaluate(args.starts, args.beam, rng)
        print(
            f"trial {t:3d} | LT={B.LT} LF={B.LF} RESERVE={B.RESERVE} "
            f"K_NEAR={B.K_NEAR} K_PER_MAT={B.K_PER_MAT} | obj={f:.4f} | {time.time() - t0:.0f}s"
        )
        if x is not None and f < overall_f:
            overall_x, overall_f = x, f
            with open("solution_tuned.json", "w") as fh:
                json.dump(
                    {
                        "decisionVector": [float(v) for v in x],
                        "problem": "mine-the-belt",
                        "challenge": "spoc-mining",
                    },
                    fh,
                )
            print(f"  *** rezultat i ri me i miri: {f:.4f} (ruajtur ne solution_tuned.json)")

    print(f"\nMe e miri nga tuning: {overall_f:.4f}  (lider: -10.687)")


if __name__ == "__main__":
    main()
