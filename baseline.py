"""
Baseline greedy per SpOC Mine the Belt.

Ideja: nisim nga nje asteroid, dhe ne cdo hap zgjedh asteroidin tjeter qe jep
"vlere / kosto" me te mire. Vlera = material i mbledhur (me peshe me te madhe per
materialin qe po ngelet pas, sepse objektivi eshte MIN i 3 materialeve) ose
karburant. Kostoja = karburanti i harxhuar per transfer + pak kosto per kohen.

Perdorimi:
    python baseline.py --runs 3 --seed 0
Krijon solution.json me vektorin e vendimeve (30000 numra).
"""
import argparse
import json
import numpy as np
import pykep as pk
import udp as U

N = U.udp.n
MU = U.MU_TRAPPIST
T0 = U.T_START.mjd2000
WINDOW = U.udp.mission_window[1]  # 1827 dite
DVF = U.DV_per_fuel
TMINE = U.TIME_TO_MINE_FULLY

MASSES = U.asteroid_masses
MATS = U.asteroid_materials

# Pool-et sipas llojit te materialit (0,1,2 = materiale, 3 = karburant)
POOLS = {m: np.where(MATS == m)[0] for m in range(4)}

TOF_GRID = [6.0, 15.0, 30.0]  # dite fluturimi qe provojme
SAMPLE_PER_POOL = 100  # sa kandidate per pool ne cdo hap
TIME_COST = 0.5  # sa "karburant-ekuivalent" kushton e gjithe dritarja kohore


def transfer_dv(a_from, t_dep, a_to, t_arr):
    """DV (m/s) per transferin nga a_from (ne t_dep) ne a_to (ne t_arr). None nese deshton."""
    try:
        r1, v1 = U.asteroids[a_from].eph(T0 + t_dep)
        r2, v2 = U.asteroids[a_to].eph(T0 + t_arr)
        l = pk.lambert_problem(
            r1=r1, r2=r2, tof=(t_arr - t_dep) * pk.DAY2SEC, mu=MU, cw=False, max_revs=0
        )
        dv1 = np.linalg.norm(np.array(v1) - np.array(l.get_v1()[0]))
        dv2 = np.linalg.norm(np.array(v2) - np.array(l.get_v2()[0]))
        return dv1 + dv2
    except Exception:
        return None


def run(seed, verbose=True):
    rng = np.random.default_rng(seed)

    # asteroidi fillestar (nuk mblidhet material prej tij, shih UDP: loop nis nga i=1)
    start = int(rng.choice(POOLS[3]))
    ids = [start]
    t_arr = [0.0]
    t_mine = [0.0]
    visited = {start}

    fuel = 1.0
    collected = np.zeros(3)
    t_free = 0.0  # koha kur largohemi nga asteroidi aktual

    while True:
        cur = ids[-1]
        best = None  # (score, cand, tof, dv, mine_time)
        lo = collected.min()

        for m in range(4):
            pool = POOLS[m]
            k = min(SAMPLE_PER_POOL, len(pool))
            cands = rng.choice(pool, size=k, replace=False)
            for c in cands:
                c = int(c)
                if c in visited:
                    continue
                mass = MASSES[c]
                mine = min(60.0, TMINE * mass)  # koha qe e mbledh krejt masen (max 60)
                gain = min(mass, mine / TMINE)
                for tof in TOF_GRID:
                    if t_free + tof > WINDOW:
                        continue
                    dv = transfer_dv(cur, t_free, c, t_free + tof)
                    if dv is None:
                        continue
                    fuel_after = fuel - dv / DVF
                    if fuel_after < 0:
                        continue
                    cost = dv / DVF + TIME_COST * (tof + mine) / WINDOW + 1e-4
                    if m == 3:
                        # karburanti vlen shume kur depozita eshte e ulet
                        eff = min(1.0, fuel_after + gain) - fuel_after
                        factor = 3.0 if fuel_after < 0.35 else 0.4
                        value = factor * eff
                    else:
                        w = (lo + 0.01) / (collected[m] + 0.01)  # 1 per materialin me te ulet
                        value = w * gain
                    score = value / cost
                    if best is None or score > best[0]:
                        best = (score, c, tof, dv, mine)

        if best is None or best[0] <= 0:
            break

        _, c, tof, dv, mine = best
        fuel -= dv / DVF
        arrive = t_free + tof
        ids.append(c)
        t_arr.append(arrive)
        t_mine.append(mine)
        visited.add(c)
        m = MATS[c]
        gain = min(MASSES[c], mine / TMINE)
        if m == 3:
            fuel = min(1.0, fuel + gain)
        else:
            collected[m] += gain
        t_free = arrive + mine

        if verbose and len(ids) % 10 == 0:
            print(
                f"  hapi {len(ids):4d} | t={t_free:7.1f} | fuel={fuel:.2f} | "
                f"mat={collected.round(3)} | min={collected.min():.3f}"
            )

    x = U.convert_to_chromosome(t_arr + t_mine + ids)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    best_x, best_f = None, 1e9
    for r in range(args.runs):
        print(f"Run {r} (seed {args.seed + r})")
        x = run(args.seed + r)
        f = U.udp.fitness(x)
        ok = f[1] == 0 and max(f[2:]) <= 0
        print(f"  -> objektivi = {f[0]:.4f} | i vlefshem: {ok}")
        if ok and f[0] < best_f:
            best_x, best_f = x, f[0]

    if best_x is None:
        print("Asnje zgjidhje e vlefshme.")
        return

    print(f"\nMe e miri: {best_f:.4f}  (lider ne platforme: -10.687)")
    U.udp.pretty(best_x)

    with open("solution.json", "w") as fh:
        json.dump(
            {
                "decisionVector": [float(v) for v in best_x],
                "problem": "mine-the-belt",
                "challenge": "spoc-mining",
            },
            fh,
        )
    print("U ruajt solution.json")


if __name__ == "__main__":
    main()
