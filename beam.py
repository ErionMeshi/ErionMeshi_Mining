"""
beam.py - Versioni 2 per SpOC Mine the Belt: beam search me fqinj te afert.

Ndryshimet kryesore kundrejt baseline.py:
  1. Kandidatet zgjidhen me nje vleresim te shpejte te DV (diferenca e shpejtesise
     orbitale + ndryshimi i planit), pra vetem asteroide "afer" ne hapesire orbitale.
  2. Per cdo kandidat provojme koha pritjeje (w) dhe fluturimi (tof), jo vetem nje tof fiks.
  3. Kohen e trajtojme si burim te kufizuar: ranking = progres - LT*koha + LF*karburanti.
  4. Beam search (mbajme B gjendjet me te mira ne cdo hap), jo greedy me 1 degë.

Perdorimi:
    python beam.py --beam 4 --starts 1      # test i shpejte
    python beam.py --beam 10 --starts 5     # me serioz
Ruan solution_beam.json
"""
import argparse
import json
import time
import numpy as np
import pykep as pk
import udp as U

MU = U.MU_TRAPPIST
T0 = U.T_START.mjd2000
WINDOW = U.udp.mission_window[1]  # 1827 dite
DVF = U.DV_per_fuel
TMINE = U.TIME_TO_MINE_FULLY
N = U.udp.n

D = U.data
A, INC, OM = D[:, 1], D[:, 3], D[:, 4]
MASSES, MATS = U.asteroid_masses, U.asteroid_materials
VC = np.sqrt(MU / A)  # shpejtesia orbitale rrethore
POOLS = {m: np.where(MATS == m)[0] for m in range(4)}

# ---- parametra te rregullueshem ----
LT = 0.0055  # vlera e nje dite (ne njesi progresi). ~ 10.7 / 1827
LF = 0.5  # vlera e nje njesie karburanti (ne njesi progresi)
W_GRID = (0.0, 10.0, 25.0, 45.0)  # dite pritje shtese pas minimit
TOF_GRID = (10.0, 20.0, 35.0, 55.0)  # dite fluturimi
K_NEAR = 40  # sa me te afert (ne total)
K_PER_MAT = 8  # + sa me te afert per cdo lloj materiali


def est_dv(cur):
    """Vleresim i shpejte i DV nga cur te te gjithe asteroidet (vektorizuar)."""
    dva = np.abs(VC[cur] - VC)
    c = np.cos(INC[cur]) * np.cos(INC) + np.sin(INC[cur]) * np.sin(INC) * np.cos(OM[cur] - OM)
    th = np.arccos(np.clip(c, -1.0, 1.0))
    return np.hypot(dva, 0.5 * (VC[cur] + VC) * th)


def candidates(cur, visited):
    est = est_dv(cur)
    est[list(visited)] = np.inf
    sel = set(np.argpartition(est, K_NEAR)[:K_NEAR].tolist())
    for m in range(4):
        pool = POOLS[m]
        k = min(K_PER_MAT, len(pool) - 1)
        sel.update(pool[np.argpartition(est[pool], k)[:k]].tolist())
    return [c for c in sel if np.isfinite(est[c])]


def transfer_dv(a_from, t_dep, a_to, t_arr):
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


def potential(coll):
    return 0.5 * coll.min() + 0.5 * coll.mean()


class S:
    __slots__ = ("ids", "ta", "tm", "fuel", "coll", "vis", "key")


def root(start):
    s = S()
    s.ids, s.ta, s.tm = [start], [0.0], [0.0]
    s.fuel, s.coll, s.vis, s.key = 1.0, np.zeros(3), {start}, 0.0
    return s


def expand(s):
    cur = s.ids[-1]
    base = 0.0 if len(s.ids) == 1 else TMINE * MASSES[cur]
    out = []
    for c in candidates(cur, s.vis):
        mass = MASSES[c]
        m = MATS[c]
        base_c = TMINE * mass
        best = None
        for w in W_GRID:
            tm = min(60.0, base + w)
            t_dep = s.ta[-1] + tm
            for tof in TOF_GRID:
                t_arr = t_dep + tof
                if t_arr > WINDOW:
                    continue
                dv = transfer_dv(cur, t_dep, c, t_arr)
                if dv is None:
                    continue
                fuel_after = s.fuel - dv / DVF
                if fuel_after < 0:
                    continue
                J = LT * (tm - base + tof) + LF * dv / DVF
                if best is None or J < best[0]:
                    best = (J, tm, t_arr, fuel_after)
        if best is None:
            continue
        _, tm, t_arr, fuel_after = best
        ch = S()
        ch.ids = s.ids + [c]
        ch.ta = s.ta + [t_arr]
        ch.tm = s.tm[:-1] + [tm, base_c]
        coll = s.coll.copy()
        fuel = fuel_after
        if m == 3:
            fuel = min(1.0, fuel + mass)
        else:
            coll[m] += mass
        ch.fuel, ch.coll, ch.vis = fuel, coll, s.vis | {c}
        ch.key = potential(coll) - LT * (t_arr + base_c) + LF * fuel
        out.append(ch)
    return out


def beam_search(start, B, verbose=True):
    beam = [root(start)]
    best, best_obj = None, -1.0
    depth = 0
    while beam:
        children = []
        for s in beam:
            children.extend(expand(s))
        if not children:
            break
        children.sort(key=lambda s: s.key, reverse=True)
        seen, nb = set(), []
        for ch in children:
            sig = (ch.ids[-1], int(ch.ta[-1] // 15))
            if sig in seen:
                continue
            seen.add(sig)
            nb.append(ch)
            if len(nb) >= B:
                break
        for ch in children:
            o = ch.coll.min()
            if o > best_obj:
                best_obj, best = o, ch
        beam = nb
        depth += 1
        if verbose and depth % 5 == 0:
            top = nb[0]
            print(
                f"  hapi {depth:3d} | t={top.ta[-1]:7.1f} | fuel={top.fuel:.2f} | "
                f"mat={top.coll.round(2)} | me i miri deri tani: {best_obj:.3f}"
            )
    return best


def to_x(s):
    return U.convert_to_chromosome(s.ta + s.tm + s.ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    best_x, best_f = None, 1e9
    for r in range(args.starts):
        start = int(rng.integers(N))
        t = time.time()
        print(f"Start {r}: asteroidi {start}, beam={args.beam}")
        s = beam_search(start, args.beam)
        x = to_x(s)
        f = U.udp.fitness(x)
        ok = f[1] == 0 and max(f[2:]) <= 0
        print(f"  -> objektivi = {f[0]:.4f} | i vlefshem: {ok} | {time.time() - t:.0f}s")
        if ok and f[0] < best_f:
            best_x, best_f = x, f[0]
            with open("solution_beam.json", "w") as fh:
                json.dump(
                    {
                        "decisionVector": [float(v) for v in best_x],
                        "problem": "mine-the-belt",
                        "challenge": "spoc-mining",
                    },
                    fh,
                )

    if best_x is None:
        print("Asnje zgjidhje e vlefshme.")
        return
    print(f"\nMe e miri: {best_f:.4f}  (lider: -10.687)")
    U.udp.pretty(best_x)
    print("U ruajt solution_beam.json")


if __name__ == "__main__":
    main()
