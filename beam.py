
"""
beam.py - Versioni 3 per SpOC Mine the Belt: beam search me fqinj te afert.
 
Ndryshimet kundrejt Versionit 2 (emrat e variablave globale mbeten te njejta,
pra skripta jote e tuning-ut punon pa ndryshime):
 
  LOGJIKE (korrigjime)
  1. RESERVE ishte e perkufizuar por nuk perdorej asgjekundi. Tani zbatohet:
     pas nje hopi te nje asteroidi materiali duhet te mbetet >= RESERVE karburant
     (nuk vlen per asteroidet e karburantit, dhe as ne END_FREE ditet e fundit).
  2. Koha e minimit te asteroidit aktual lexohet nga s.tm[-1] (nje burim i vetem
     e vertete), pa rastin e vecante "if len(ids)==1".
  3. Nje asteroid kandidat pranohet vetem nese minimi i plote perfundon brenda
     dritares (t_arr + koha_e_minimit <= WINDOW) dhe nese koha e plote e minimit
     nuk e kalon kufirin MAX_MINE (para ketij ndryshimi kreditohej mase e plote
     edhe kur minimi s'kishte kohe te perfundonte).
  4. Zgjidhja me e mire zgjidhet me (min, mesatare) dhe kontrollohet me
     U.udp.fitness mbi FINALISTS kandidate, jo vetem mbi nje.
  5. solution_beam.json nuk mbishkruhet me nje rezultat me te keq se ai qe
     ekziston tashme ne skedar.
 
  ALGORITM
  6. Kandidatet varen nga nevoja: materiali me i pakti i mbledhur (min) merr
     K_BONUS kandidate shtese; po ashtu karburanti kur fuel < LOW_FUEL.
  7. Efikasitet: pozicioni i nisjes llogaritet nje here per (w), efemeridat ruhen
     ne cache (lru_cache) -> shume me pak thirrje eph.
  8. Lambert mund te provoje edhe zgjidhje me rrotullime (REVS / --revs).
  9b. REFINE (--refine-secs): pas beam-it, rifuton pjese te fundit te rruges
     (nga nje pike e rastesishme) me beam te ri dhe parametra pak te perturbuar;
     pranon vetem nese U.udp.fitness permireson.
  9. Paralelizim mbi start-et (--jobs), start-e te caktuara (--start),
     minim i asteroidit fillestar (--mine-start, ofrohet si opsion).
 
Perdorimi:
    python beam.py --beam 4 --starts 1              # test i shpejte
    python beam.py --beam 10 --starts 8 --jobs 4    # me serioz, paralel
    python beam.py --beam 10 --start 123 456        # start-e te caktuara
Ruan solution_beam.json (vetem nese permireson rezultatin ekzistues).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from heapq import nlargest
 
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
 
# ---- parametra te rregullueshem (te njejtat si ne v2 + te rinj) ----
LT = 0.0055  # vlera e nje dite (ne njesi progresi). ~ 10.7 / 1827
LF = 0.5  # vlera e nje njesie karburanti (ne njesi progresi)
W_GRID = (0.0, 10.0, 25.0, 45.0)  # dite pritje shtese pas minimit
TOF_GRID = (10.0, 20.0, 35.0, 55.0)  # dite fluturimi
RESERVE = 0.15  # karburant minimal pas hopit te nje asteroidi materiali
END_FREE = 120.0  # ne kete numer ditesh te fundit RESERVE nuk zbatohet
K_NEAR = 40  # sa me te afert (ne total)
K_PER_MAT = 16  # + sa me te afert per cdo lloj materiali
K_BONUS = 8  # kandidate shtese per materialin me te pakte / karburantin kur mungon
LOW_FUEL = 0.35  # nen kete nivel karburanti, asteroidet e karburantit marrin bonus
ALPHA = 0.5  # potential = ALPHA*min + (1-ALPHA)*mesatare
REVS = 0  # sa rrotullime maksimale per Lambert (0 = si ne v2)
MINE_START = False  # nese True, minohet edhe asteroidi fillestar
DEDUP_DAYS = 15.0  # gjerësia e kovës kohore per heqjen e dublikateve
FINALISTS = 8  # sa zgjidhje finaliste kontrollohen me fitness te vertete
MAX_MINE = 60.0  # kufiri i kohes se minimit ne nje asteroid (dite)
MASS_POW = 0.0  # kandidatet renditen me est_dv / masa**MASS_POW (0 = vetem DV, si ne v2)
MIN_MASS = 0.0  # asteroidet e materialit (0-2) me masë nën kete prag perjashtohen (0 = çaktivizuar)
ENDGAME_DAYS = 0.0  # ne kete numer ditesh te fundit: vetem materiali me i pakti, ALPHA=1 (0 = çaktivizuar)
 
PARAM_KEYS = (
    "LT", "LF", "W_GRID", "TOF_GRID", "RESERVE", "END_FREE", "K_NEAR", "K_PER_MAT",
    "K_BONUS", "LOW_FUEL", "ALPHA", "REVS", "MINE_START", "DEDUP_DAYS", "FINALISTS",
    "MAX_MINE", "MASS_POW", "MIN_MASS", "ENDGAME_DAYS",
)
 
 
def set_cfg(cfg):
    """Vendos parametrat globale (perdoret edhe si initializer i proceseve punetore)."""
    globals().update(cfg)
 
 
def valid(f):
    return f[1] == 0 and max(f[2:]) <= 0
 
 
# ---------------------------------------------------------------- fizika
def est_dv(cur):
    """Vleresim i shpejte i DV nga cur te te gjithe asteroidet (vektorizuar)."""
    dva = np.abs(VC[cur] - VC)
    c = np.cos(INC[cur]) * np.cos(INC) + np.sin(INC[cur]) * np.sin(INC) * np.cos(OM[cur] - OM)
    th = np.arccos(np.clip(c, -1.0, 1.0))
    return np.hypot(dva, 0.5 * (VC[cur] + VC) * th)
 
 
@lru_cache(maxsize=200_000)
def _eph(a, t):
    return U.asteroids[a].eph(T0 + t)
 
 
def lambert_dv(r1, v1, r2, v2, tof):
    """DV minimal (dv1 + dv2) mes te gjitha zgjidhjeve te Lambert-it, ose None."""
    try:
        l = pk.lambert_problem(
            r1=r1, r2=r2, tof=tof * pk.DAY2SEC, mu=MU, cw=False, max_revs=REVS
        )
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        best = None
        for a, b in zip(l.get_v1(), l.get_v2()):
            dv = np.linalg.norm(v1 - np.asarray(a)) + np.linalg.norm(v2 - np.asarray(b))
            if best is None or dv < best:
                best = dv
        return best
    except Exception:
        return None
 
 
def transfer_dv(a_from, t_dep, a_to, t_arr):
    """Pajtueshmeri me v2: DV per nje transfer te vetem."""
    r1, v1 = _eph(a_from, t_dep)
    r2, v2 = _eph(a_to, t_arr)
    return lambert_dv(r1, v1, r2, v2, t_arr - t_dep)
 
 
# ---------------------------------------------------------------- kandidatet
def candidates(cur, visited, coll=None, fuel=1.0, t_now=0.0):
    """K_NEAR me te afertit + K_PER_MAT per material; me shume per nevojen aktuale."""
    est = est_dv(cur)
    est[list(visited)] = np.inf
    if MIN_MASS:  # vetem asteroide materiali te rende mjaftueshem
        est[(MASSES < MIN_MASS) & (MATS != 3)] = np.inf
    if ENDGAME_DAYS and coll is not None and t_now > WINDOW - ENDGAME_DAYS:
        # ne fund cdo hop duhet te rrise materialin me te pakte (objektivi eshte min)
        est[(MATS != int(np.argmin(coll))) & (MATS != 3)] = np.inf
    if MASS_POW:  # preferon asteroide me mase te madhe (amortizojne koston e hopit)
        est = est / np.maximum(MASSES, 1e-6) ** MASS_POW
    sel = set(np.argpartition(est, K_NEAR)[:K_NEAR].tolist())
    need = int(np.argmin(coll)) if coll is not None else -1
    for m in range(4):
        pool = POOLS[m]
        k = K_PER_MAT
        if m == need or (m == 3 and fuel < LOW_FUEL):
            k += K_BONUS
        k = min(k, len(pool) - 1)
        sel.update(pool[np.argpartition(est[pool], k)[:k]].tolist())
    return [c for c in sel if np.isfinite(est[c])]
 
 
def potential(coll, alpha=None):
    a = ALPHA if alpha is None else alpha
    return a * coll.min() + (1.0 - a) * coll.mean()
 
 
# ---------------------------------------------------------------- gjendja
class S:
    __slots__ = ("ids", "ta", "tm", "fuel", "coll", "vis", "key", "par")
 
 
def root(start):
    s = S()
    s.ids, s.ta, s.vis, s.key, s.par = [start], [0.0], {start}, 0.0, None
    s.fuel, s.coll, s.tm = 1.0, np.zeros(3), [0.0]
    if MINE_START:
        t_mine = TMINE * MASSES[start]
        if t_mine <= MAX_MINE:
            s.tm = [t_mine]
            if MATS[start] != 3:
                s.coll[MATS[start]] += MASSES[start]
    return s
 
 
def expand(s):
    cur = s.ids[-1]
    base = s.tm[-1]  # koha minimale e minimit ne cur (tashme e regjistruar)
    t_now = s.ta[-1]
 
    # nisjet e mundshme nga cur (varen vetem nga w) -> llogariten nje here
    deps = {}
    for w in W_GRID:
        tm = min(MAX_MINE, base + w)
        if tm not in deps:
            t_dep = t_now + tm
            r1, v1 = _eph(cur, t_dep)
            deps[tm] = (tm, t_dep, r1, v1)
 
    out = []
    for c in candidates(cur, s.vis, s.coll, s.fuel, t_now):
        mass, m = MASSES[c], MATS[c]
        base_c = TMINE * mass
        if base_c > MAX_MINE:  # nuk mund te minohet plotesisht
            continue
        best = None
        for tm, t_dep, r1, v1 in deps.values():
            for tof in TOF_GRID:
                t_arr = t_dep + tof
                if t_arr + base_c > WINDOW:  # minimi i plote duhet te perfundoje
                    continue
                r2, v2 = _eph(c, t_arr)
                dv = lambert_dv(r1, v1, r2, v2, tof)
                if dv is None:
                    continue
                fuel_after = s.fuel - dv / DVF
                reserve = 0.0 if (m == 3 or t_arr > WINDOW - END_FREE) else RESERVE
                if fuel_after < reserve:
                    continue
                J = LT * (tm - base + tof) + LF * dv / DVF
                if best is None or J < best[0]:
                    best = (J, tm, t_arr, fuel_after)
        if best is None:
            continue
        _, tm, t_arr, fuel_after = best
        ch = S()
        ch.par = s
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
        alpha = 1.0 if (ENDGAME_DAYS and t_arr > WINDOW - ENDGAME_DAYS) else None
        ch.key = potential(coll, alpha) - LT * (t_arr + base_c) + LF * fuel
        out.append(ch)
    return out
 
 
def _score(s):
    return (s.coll.min(), s.coll.mean(), -s.ta[-1])
 
 
def to_x(s):
    return U.convert_to_chromosome(s.ta + s.tm + s.ids)
 
 
def beam_search(start, B, verbose=True, init=None):
    beam = [init if init is not None else root(start)]
    finalists = []
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
            sig = (ch.ids[-1], int(ch.ta[-1] // DEDUP_DAYS))
            if sig in seen:
                continue
            seen.add(sig)
            nb.append(ch)
            if len(nb) >= B:
                break
        finalists = nlargest(FINALISTS, finalists + children, key=_score)
        beam = nb
        depth += 1
        if verbose and depth % 5 == 0:
            top = nb[0]
            print(
                f"  hapi {depth:3d} | t={top.ta[-1]:7.1f} | fuel={top.fuel:.2f} | "
                f"mat={top.coll.round(2)} | me i miri deri tani: {finalists[0].coll.min():.3f}"
            )
 
    if not finalists:
        return None
    # kontrollo finalistet me fitness te vertete; merr me te miren e vlefshme
    best, best_f = None, 1e9
    for s in finalists:
        f = U.udp.fitness(to_x(s))
        if valid(f) and f[0] < best_f:
            best, best_f = s, f[0]
    return best if best is not None else finalists[0]
 
 
# ---------------------------------------------------------------- rafinim
def chain_of(s):
    """Gjendjet nga rrenja deri te s (cdo paraardhes eshte nje gjendje e vlefshme)."""
    out = []
    while s is not None:
        out.append(s)
        s = s.par
    return out[::-1]
 
 
def _eval(s):
    f = U.udp.fitness(to_x(s))
    return bool(valid(f)), float(f[0])
 
 
def refine(s, B, secs, seed, verbose=True, ckpt=None):
    """Rifuton fundin e rruges nga nje pike e rastesishme me parametra te perturbuar."""
    rng = np.random.default_rng(seed)
    ok, cur_f = _eval(s)
    if not ok:
        return s
    base = {k: globals()[k] for k in ("LT", "LF", "MASS_POW", "ALPHA")}
    t_end = time.time() + secs
    rounds = 0
    while time.time() < t_end:
        chain = chain_of(s)
        if len(chain) < 3:
            break
        k = int(rng.integers(1, len(chain) - 1))
        globals().update(
            LT=base["LT"] * rng.uniform(0.7, 1.4),
            LF=base["LF"] * rng.uniform(0.6, 1.6),
            MASS_POW=max(0.0, base["MASS_POW"] + float(rng.choice([-0.5, 0.0, 0.5]))),
            ALPHA=min(0.95, max(0.2, base["ALPHA"] + rng.uniform(-0.15, 0.15))),
        )
        try:
            cand = beam_search(None, B, False, init=chain[k])
        finally:
            globals().update(base)
        rounds += 1
        if cand is None:
            continue
        ok, f = _eval(cand)
        if not ok:
            continue
        better = f < cur_f - 1e-9
        plateau = abs(f - cur_f) <= 1e-9 and cand.coll.mean() > s.coll.mean() + 1e-9
        if better or plateau:
            if better and verbose:
                print(f"  rafinim #{rounds}: prerje ne hapin {k} -> {f:.4f}")
            s, cur_f = cand, f
            if better and ckpt:
                save_if_better([float(v) for v in to_x(s)], f, ckpt)
    return s
 
 
# ---------------------------------------------------------------- ekzekutimi
def run_start(job):
    start, B, verbose, refine_secs, refine_beam, ckpt = job
    t = time.time()
    s = beam_search(start, B, verbose)
    if s is None:
        return start, None, None, False, time.time() - t
    if refine_secs > 0:
        s = refine(s, refine_beam or B, refine_secs, start, verbose, ckpt)
    x = to_x(s)
    f = U.udp.fitness(x)
    return start, [float(v) for v in x], [float(v) for v in f], bool(valid(f)), time.time() - t
 
 
def load_best(path):
    """Rezultati i vlefshem ekzistues ne skedar (x, objektivi) ose (None, 1e9)."""
    if os.path.exists(path):
        try:
            with open(path) as fh:
                x = json.load(fh)["decisionVector"]
            f = U.udp.fitness(x)
            if valid(f):
                return x, f[0]
        except Exception:
            pass
    return None, 1e9
 
 
def save_if_better(x, f0, path):
    _, old = load_best(path)
    if f0 < old:
        with open(path, "w") as fh:
            json.dump(
                {"decisionVector": x, "problem": "mine-the-belt", "challenge": "spoc-mining"},
                fh,
            )
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--start", type=int, nargs="+", help="asteroide fillestare te caktuar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="procese paralele")
    ap.add_argument("--out", default="solution_beam.json")
    ap.add_argument("--lt", type=float, default=LT)
    ap.add_argument("--lf", type=float, default=LF)
    ap.add_argument("--reserve", type=float, default=RESERVE)
    ap.add_argument("--end-free", type=float, default=END_FREE)
    ap.add_argument("--k-near", type=int, default=K_NEAR)
    ap.add_argument("--k-mat", type=int, default=K_PER_MAT)
    ap.add_argument("--k-bonus", type=int, default=K_BONUS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--revs", type=int, default=REVS)
    ap.add_argument("--w", type=float, nargs="+", default=list(W_GRID))
    ap.add_argument("--tof", type=float, nargs="+", default=list(TOF_GRID))
    ap.add_argument("--mine-start", action="store_true")
    ap.add_argument("--finalists", type=int, default=FINALISTS)
    ap.add_argument("--mass-pow", type=float, default=MASS_POW)
    ap.add_argument("--min-mass", type=float, default=MIN_MASS)
    ap.add_argument("--endgame", type=float, default=ENDGAME_DAYS)
    ap.add_argument("--refine-secs", type=float, default=0.0, help="sekonda rafinimi per start")
    ap.add_argument("--refine-beam", type=int, default=0, help="beam per rafinim (0 = --beam)")
    args = ap.parse_args()
 
    cfg = {
        "LT": args.lt, "LF": args.lf, "RESERVE": args.reserve, "END_FREE": args.end_free,
        "K_NEAR": args.k_near, "K_PER_MAT": args.k_mat, "K_BONUS": args.k_bonus,
        "ALPHA": args.alpha, "REVS": args.revs, "W_GRID": tuple(args.w),
        "TOF_GRID": tuple(args.tof), "MINE_START": args.mine_start,
        "FINALISTS": args.finalists, "MASS_POW": args.mass_pow,
        "MIN_MASS": args.min_mass, "ENDGAME_DAYS": args.endgame,
    }
    set_cfg(cfg)
 
    if args.start:
        starts = list(args.start)
    else:
        rng = np.random.default_rng(args.seed)
        starts = [int(rng.integers(N)) for _ in range(args.starts)]
 
    best_x, best_f = load_best(args.out)
    if best_x is not None:
        print(f"Rezultati ekzistues ne {args.out}: {best_f:.4f}")
    new_best = False
 
    verbose = args.jobs == 1
    ckpt = args.out if args.jobs == 1 else None
    jobs = [
        (st, args.beam, verbose, args.refine_secs, args.refine_beam, ckpt) for st in starts
    ]
    ex = None
    if args.jobs > 1:
        ex = ProcessPoolExecutor(max_workers=args.jobs, initializer=set_cfg, initargs=(cfg,))
        results = ex.map(run_start, jobs)
    else:
        results = map(run_start, jobs)
 
    for r, (start, x, f, ok, secs) in enumerate(results):
        print(f"Start {r}: asteroidi {start}, beam={args.beam}")
        if x is None:
            print("  -> nuk u gjet asnje rruge")
            continue
        print(f"  -> objektivi = {f[0]:.4f} | i vlefshem: {ok} | {secs:.0f}s")
        if ok and f[0] < best_f:
            best_x, best_f, new_best = x, f[0], True
            with open(args.out, "w") as fh:
                json.dump(
                    {
                        "decisionVector": best_x,
                        "problem": "mine-the-belt",
                        "challenge": "spoc-mining",
                    },
                    fh,
                )
 
    if ex is not None:
        ex.shutdown()
 
    if best_x is None:
        print("Asnje zgjidhje e vlefshme.")
        return
    print(f"\nMe e miri: {best_f:.4f}  (lider: -10.687)")
    U.udp.pretty(best_x)
    print(f"U ruajt {args.out}" if new_best else f"{args.out} mbeti i pandryshuar (nuk u permiresua)")
 
 
if __name__ == "__main__":
    main()

