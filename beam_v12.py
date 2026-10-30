"""
beam_v12.py - Versioni 12 per SpOC Mine the Belt (testuar mbi candidates.txt te vertete).

Ndryshimet kundrejt v11 (v12): te gjitha kombinimet (w, tof) te screening-ut dhe te refinimit grupohen ne NJE thirrje Lambert
(shpenzimi dominues ishte overhead-i i numpy-t per thirrje te vogla).

Ndryshimet kundrejt v10 (v11): SHPEJTESI ~3x.  (1) grid pozicionesh/shpejtesish (float32, hap 1 dite, interpolim
Hermite) per pos_proxy, qe e zevendeson kepler_state mbi 10000 asteroide; (2) lookahead-et (gamma, kappa) llogariten
leneshem vetem per 3B gjendjet me te mira te koshes; (3) refinim me me pak kombinime dhe iters=32.

Ndryshimet kundrejt v9 (v10): LOOKAHEAD I RI (kappa): kur nje kose kohore hapet, per 3B gjendjet me te mira llogaritet
sa te lira jane hopet e ardhshme (mesatarja e 3 vlerave me te vogla te pos_proxy mbi asteroidet e renda te pavizituar)
dhe rankingu korrigjohet me  -kappa*LA/DVF.  Kjo favorizon gjendjet qe ndodhen ne zona me fqinje te lire.

Ndryshimet kundrejt v8 (v9): KANDIDATET ZGJIDHEN ME PROXY POZICIONI (pos_proxy), jo me est_dv.
Testi mbi te dhenat reale tregoi qe est_dv (diferenca e shpejtesive rrethore + ndryshimi i planit) kap
vetem ~22% te 20 hopeve me te lira te vertete brenda 150 kandidateve, ndersa pos_proxy
(distanca e pozicioneve ne t+tof / tof, + shpejtesia relative) kap ~91% brenda 150 dhe ~98% brenda 300.

Ndryshimet kundrejt v7 (v8): kosha kohore sipas kohes se GATSHMERISE (mberritje + minim), jo vetem mberritjes
(--bready 1), qe gjendjet qe krahasohen te jene gati per nisje afersisht ne te njejten kohe.

Ndryshimet kundrejt v6:
  1. SHPEJTESI: refinimi i kandidateve behet i vektorizuar (Lambert per te gjithe topm ne nje thirrje
     per cdo (w, tof)), jo me thirrje skalare pykep. Ne verifikim perdoret vetem U.udp.fitness.
  2. LAMBERT per cifte: lambert_dv pranon edhe nisje/koha te ndryshme per cdo target.
  3. KOSTO REALE E KARBURANTIT: zgjedhja e (w, tof) per cdo hop minimizon  dite + LAM*dv/DVF
     (LAM = dite ekuivalente per nje njesi karburanti), ndersa rankingu i beam-it perdor LT/LF.
  4. MASA: asteroidet me mase te vogel perjashtohen (mmin per mat 0/2, mmin1 per mat 1, fmin per
     karburantin) - nje hop ka kosto fikse, prandaj te lehtat humbin kohe.
  5. KARBURANTI NE FUND: vlera e karburantit kufizohet nga BURN*(dite te mbetura).
  6. MARGJINE SIGURIE: karburanti pas cdo hopi >= EPS (1e-4), qe ndryshimet numerike pykep te mos
     e prishin zgjidhjen.
  7. CLI automatik per cdo parameter skalar te DEFAULT_P.

Perdorimi:
    python beam_v12.py --beam 8 --starts 3 --stratified --jobs 4
    python beam_v12.py --start-id 8655 --beam 16 --rounds 6 --jobs 4
"""
import argparse
import heapq
import itertools
import json
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pykep as pk
import udp as U

MU = U.MU_TRAPPIST
T0 = U.T_START.mjd2000
WINDOW = U.udp.mission_window[1]
DVF = U.DV_per_fuel
TMINE = U.TIME_TO_MINE_FULLY
N = U.udp.n
TWO_PI = 2.0 * np.pi

D = U.data
A, ECC, INC, OM, ARG, M0 = D[:, 1], D[:, 2], D[:, 3], D[:, 4], D[:, 5], D[:, 6]
MASSES, MATS = U.asteroid_masses, U.asteroid_materials
VC = np.sqrt(MU / A)
NMEAN = np.sqrt(MU / A**3)
POOLS = {m: np.where(MATS == m)[0] for m in range(4)}

DEFAULT_P = dict(
    kappa=0.0,  # pesha e lookahead-it te ri (0 = i fikur; ne teste nuk dha permiresim)
    la_mass=0.6,  # masa minimale e asteroideve qe numerohen ne lookahead
    screen_pm=25,  # kandidate shtese per cdo material sipas proxy pozicioni
    pw2=10.0,  # proxy llogaritet edhe per nisje te vonuar me kaq dite
    bready=1,  # 1: kosha kohore sipas kohes se gatshmerise (mberritje + minim)
    lt=0.004,  # vlera e nje dite (progres)
    lam=50.0,  # dite ekuivalente per nje njesi karburanti (per zgjedhjen e (w, tof))
    mmin=0.35,  # masa minimale e asteroideve te materialit 0/2
    mmin1=0.15,  # masa minimale e asteroideve te materialit 1 (te rralle)
    fmin=0.40,  # masa minimale e asteroideve te karburantit
    burn=0.006,  # konsum i pritur karburanti (njesi/dite) per vleresimin ne fund
    eps=1e-4,  # margjine sigurie e karburantit pas cdo hopi
    refw=(-5.0, 0.0, 5.0),
    reftof=(0.75, 0.9, 1.0, 1.15, 1.35),
    lf=0.3,  # vlera e nje njesie karburanti
    alpha=0.2,  # alpha0: pesha e min-it ne fillim (rritet drejt 1)
    alpha_pow=2.0,  # sa shpejt rritet pesha e min-it me kohen
    gamma=0.5,  # pesha e lookahead-it
    reserve=0.15,  # karburant minimal pas hopit te materialit
    refuel_to=1.0,  # deri ku e mbushim karburantin kur minojme propellant
    # --- screening fazor ---
    screen=200,  # sa asteroide te afert (pa faze) hyjne ne screening
    topm=30,  # sa kandidate verifikohen me pykep
    wscreen=(0.0, 10.0),
    tofscreen=(3.0, 6.0, 10.0, 16.0, 24.0, 36.0, 52.0),
    # --- fallback grid (v4) ---
    k_near=40,
    k_per_mat=16,
    k_need=16,
    k_fuel=10,
    k_mass=10,
    wgrid=(0.0, 10.0, 25.0, 45.0),
    tofgrid=(10.0, 20.0, 35.0, 55.0),
)
REFINE_R = 6


# ================================================================== fizika (pykep)
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


# ================================================================== fizika (vektorizuar)
def kepler_state(idx, t):
    """Pozicioni/shpejtesia e asteroideve idx ne kohen t (dite nga T_START). Kthen (r, v), shape (m,3)."""
    idx = np.asarray(idx)
    a, e = A[idx], ECC[idx]
    M = np.mod(M0[idx] + NMEAN[idx] * (t * 86400.0), TWO_PI)
    E = M + e * np.sin(M)
    for _ in range(8):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    cE, sE = np.cos(E), np.sin(E)
    b = np.sqrt(1.0 - e * e)
    r = a * (1.0 - e * cE)
    x, y = a * (cE - e), a * b * sE
    k = np.sqrt(MU * a) / r
    vx, vy = -k * sE, k * b * cE
    cw_, sw_ = np.cos(ARG[idx]), np.sin(ARG[idx])
    ci, si = np.cos(INC[idx]), np.sin(INC[idx])
    cO, sO = np.cos(OM[idx]), np.sin(OM[idx])

    def rot(px, py):
        xp = px * cw_ - py * sw_
        yp = px * sw_ + py * cw_
        y1, z1 = yp * ci, yp * si
        return np.stack([xp * cO - y1 * sO, xp * sO + y1 * cO, z1], axis=1)

    return rot(x, y), rot(vx, vy)


def _stumpff(z):
    small = np.abs(z) < 1e-6
    zs = np.where(small, 1.0, z)
    sq = np.sqrt(np.abs(zs))
    pos = zs > 0
    C = np.where(pos, (1.0 - np.cos(sq)) / zs, (np.cosh(sq) - 1.0) / (-zs))
    S = np.where(pos, (sq - np.sin(sq)) / sq**3, (np.sinh(sq) - sq) / sq**3)
    return np.where(small, 0.5, C), np.where(small, 1.0 / 6.0, S)


def lambert_dv(r1, v1, r2, v2, tau_days, iters=26):
    """DV (inf ku deshton) i transferit prograde 0-rev. r1,v1: (3,) ose (m,3); tau_days: skalar ose (m,).
    Universal variables (Curtis) me bisektim, e vektorizuar."""
    with np.errstate(all="ignore"):
        r1 = np.atleast_2d(r1)
        v1 = np.atleast_2d(v1)
        tau = np.asarray(tau_days, dtype=float) * 86400.0
        m = len(r2)
        r1n = np.linalg.norm(r1, axis=1)
        r2n = np.linalg.norm(r2, axis=1)
        cosd = np.einsum("ij,ij->i", r2, np.broadcast_to(r1, (m, 3))) / (r1n * r2n)
        crossz = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
        sind = np.sqrt(np.clip(1.0 - cosd**2, 0.0, None)) * np.where(crossz >= 0, 1.0, -1.0)
        Ac = sind * np.sqrt(r1n * r2n / np.maximum(1.0 - cosd, 1e-12))
        sm = np.sqrt(MU)

        def F(z):
            C, S = _stumpff(z)
            y = r1n + r2n + Ac * (z * S - 1.0) / np.sqrt(C)
            yy = np.where(y > 0, y, 1.0)
            val = (yy / C) ** 1.5 * S + Ac * np.sqrt(yy) - sm * tau
            return np.where(y > 0, val, -1.0), y

        lo = np.full(m, -60.0)
        hi = np.full(m, TWO_PI**2 - 1e-3)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f, _y = F(mid)
            neg = f < 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        z = 0.5 * (lo + hi)
        f, y = F(z)
        valid = (y > 0) & (np.abs(f) < 1e-3 * sm * tau) & (np.abs(Ac) > 1e-9)
        fg = 1.0 - y / r1n
        g = Ac * np.sqrt(y / MU)
        gd = 1.0 - y / r2n
        vl1 = (r2 - fg[:, None] * r1) / g[:, None]
        vl2 = (gd[:, None] * r2 - r1) / g[:, None]
        dv = np.linalg.norm(v1 - vl1, axis=1) + np.linalg.norm(v2 - vl2, axis=1)
        return np.where(valid & np.isfinite(dv), dv, np.inf)


def _validate_phase():
    """Krahaso propagimin dhe Lambert-in e vektorizuar me pykep. True vetem nese perputhen."""
    try:
        rng = np.random.default_rng(123)
        ids = rng.integers(N, size=12)
        for i in ids[:4]:
            r, v = kepler_state([i], 137.0)
            r0, v0 = U.asteroids[int(i)].eph(T0 + 137.0)
            if np.linalg.norm(r[0] - np.array(r0)) > 1e-5 * np.linalg.norm(r0):
                return False
            if np.linalg.norm(v[0] - np.array(v0)) > 1e-5 * np.linalg.norm(v0):
                return False
        cur = int(ids[0])
        tgt = ids[1:]
        r1, v1 = kepler_state([cur], 50.0)
        r2, v2 = kepler_state(tgt, 70.0)
        dvv = lambert_dv(r1[0], v1[0], r2, v2, 20.0)
        errs = []
        for k, c in enumerate(tgt):
            ex = transfer_dv(cur, 50.0, int(c), 70.0)
            if ex is not None and np.isfinite(dvv[k]):
                errs.append(abs(ex - dvv[k]) / max(ex, 1.0))
        return len(errs) >= 4 and float(np.median(errs)) < 1e-3
    except Exception:
        return False


PHASE_OK = _validate_phase()


# ================================================================== kandidate (fallback v4)
ALLIDX = np.arange(N)
PTOFS = (4.0, 8.0, 12.0, 18.0, 27.0)
GRID_MAX = 2000
_GP = None
_GV = None


def ensure_grid():
    """Ndertoje (nje here) grid-in e pozicioneve/shpejtesive te te gjithe asteroideve, hap 1 dite (float32)."""
    global _GP, _GV
    if _GP is not None:
        return
    gp = np.empty((GRID_MAX, N, 3), dtype=np.float32)
    gv = np.empty((GRID_MAX, N, 3), dtype=np.float32)
    for k in range(GRID_MAX):
        r, v = kepler_state(ALLIDX, float(k))
        gp[k] = r
        gv[k] = v
    _GP, _GV = gp, gv


def grid_pv(t):
    """Pozicion (Hermite kubik) dhe shpejtesi (linear) per te gjithe asteroidet ne kohen t, nga grid-i."""
    t = min(max(t, 0.0), GRID_MAX - 1.001)
    i = int(t)
    f = t - i
    h = 86400.0
    h00 = 2 * f**3 - 3 * f**2 + 1
    h10 = f**3 - 2 * f**2 + f
    h01 = -2 * f**3 + 3 * f**2
    h11 = f**3 - f**2
    p = h00 * _GP[i] + (h10 * h) * _GV[i] + h01 * _GP[i + 1] + (h11 * h) * _GV[i + 1]
    v = (1.0 - f) * _GV[i] + f * _GV[i + 1]
    return p, v


ensure_grid()


def pos_proxy(cur, t_dep):
    """Proxy fazor i DV (m/s) nga cur (nisje ne t_dep) per te gjithe asteroidet: distanca e pozicioneve ne
    t_dep+tof pjestuar me tof, e kombinuar me shpejtesine relative; minimumi mbi disa tof."""
    best = np.full(N, np.inf)
    for tf in PTOFS:
        rc, vc = kepler_state([cur], t_dep + tf)
        rt, vt = grid_pv(t_dep + tf)
        d = np.linalg.norm(rt - rc, axis=1) / (tf * 86400.0)
        dvv = np.linalg.norm(vt - vc, axis=1)
        best = np.minimum(best, np.hypot(d, 0.5 * dvv))
    return best


def est_dv(cur, idx=None):
    if idx is None:
        vc, inc, om = VC, INC, OM
    else:
        vc, inc, om = VC[idx], INC[idx], OM[idx]
    dva = np.abs(VC[cur] - vc)
    c = np.cos(INC[cur]) * np.cos(inc) + np.sin(INC[cur]) * np.sin(inc) * np.cos(OM[cur] - om)
    th = np.arccos(np.clip(c, -1.0, 1.0))
    return np.hypot(dva, 0.5 * (VC[cur] + vc) * th)


HEAVY = np.flatnonzero(MASSES >= 0.6)


def near_cost(s):
    """Mesatarja e 3 vlerave me te vogla te pos_proxy (m/s) nga gjendja s drejt asteroideve te renda te pavizituar."""
    cur = s.ids[-1]
    t_dep = s.ta[-1] + min(60.0, s.tm[-1])
    heavy = HEAVY[~np.isin(HEAVY, np.fromiter(s.vis, dtype=int, count=len(s.vis)))]
    best = np.full(len(heavy), np.inf)
    for tf in (8.0, 14.0):
        rc, vc = kepler_state([cur], t_dep + tf)
        rt, vt = grid_pv(t_dep + tf)
        rt, vt = rt[heavy], vt[heavy]
        d = np.linalg.norm(rt - rc, axis=1) / (tf * 86400.0)
        dvv = np.linalg.norm(vt - vc, axis=1)
        best = np.minimum(best, np.hypot(d, 0.5 * dvv))
    return float(np.partition(best, 2)[:3].mean())


def lookahead(c, coll, vis, fuel):
    pool = POOLS[3] if fuel < 0.3 else POOLS[int(np.argmin(coll))]
    e = est_dv(c, pool)
    e[np.isin(pool, np.fromiter(vis, dtype=int, count=len(vis)))] = np.inf
    k = min(3, len(e))
    near = np.partition(e, k - 1)[:k]
    near = near[np.isfinite(near)]
    return float(near.mean()) if len(near) else float(VC.mean())


def _take(est, idx, k):
    k = min(k, len(idx))
    if k <= 0:
        return []
    if k == len(idx):
        return idx.tolist()
    return idx[np.argpartition(est[idx], k - 1)[:k]].tolist()


def candidates(cur, s, P):
    est = est_dv(cur)
    est[list(s.vis)] = np.inf
    est[est > 2.0 * s.fuel * DVF] = np.inf
    allidx = np.arange(N)
    sel = set(_take(est, allidx[np.isfinite(est)], P["k_near"]))
    for m in range(4):
        pool = POOLS[m]
        pool = pool[np.isfinite(est[pool])]
        sel.update(_take(est, pool, P["k_per_mat"]))
    pool = POOLS[int(np.argmin(s.coll))]
    sel.update(_take(est, pool[np.isfinite(est[pool])], P["k_need"]))
    if s.fuel < 0.35:
        pool = POOLS[3]
        sel.update(_take(est, pool[np.isfinite(est[pool])], P["k_fuel"]))
    fin = allidx[np.isfinite(est)]
    if len(fin):
        eff = est[fin] / np.sqrt(MASSES[fin])
        k = min(P["k_mass"], len(fin))
        sel.update(fin[np.argpartition(eff, k - 1)[:k]].tolist())
    return [c for c in sel if np.isfinite(est[c])]


# ================================================================== gjendja
class S:
    __slots__ = ("ids", "ta", "tm", "fuel", "coll", "vis", "key", "parent", "look", "look2")

    def __getstate__(self):
        return (self.ids, self.ta, self.tm, self.fuel, self.coll, self.vis, self.key)

    def __setstate__(self, st):
        self.ids, self.ta, self.tm, self.fuel, self.coll, self.vis, self.key = st
        self.parent = None
        self.look = None
        self.look2 = None


def root(start):
    s = S()
    s.ids, s.ta, s.tm = [start], [0.0], [0.0]
    s.fuel, s.coll, s.vis, s.key, s.parent, s.look, s.look2 = 1.0, np.zeros(3), {start}, 0.0, None, None, None
    return s


def alpha_t(P, t):
    a0 = P["alpha"]
    return a0 + (1.0 - a0) * (min(max(t, 0.0), WINDOW) / WINDOW) ** P["alpha_pow"]


def potential(coll, P, t):
    a = alpha_t(P, t)
    return a * coll.min() + (1.0 - a) * coll.mean()


def best_leg(s, cur, c, base, P, ws, tofs):
    """Grid (w, tof) me pykep -> (J, tm, t_arr, dv) me J minimale, ose None."""
    best = None
    for w in ws:
        tm = min(60.0, base + w)
        t_dep = s.ta[-1] + tm
        for tof in tofs:
            t_arr = t_dep + tof
            if t_arr > WINDOW or tof < 0.5:
                continue
            dv = transfer_dv(cur, t_dep, c, t_arr)
            if dv is None or s.fuel - dv / DVF < P["eps"]:
                continue
            if MATS[c] != 3 and s.fuel - dv / DVF < P["reserve"]:
                continue
            J = P["lt"] * (tm - base + tof) + P["lf"] * dv / DVF
            if best is None or J < best[0]:
                best = (J, tm, t_arr, dv)
    return best


def make_child(s, c, tm, t_arr, dv, P):
    mass, m = MASSES[c], MATS[c]
    fuel = s.fuel - dv / DVF
    full = min(60.0, TMINE * mass)
    coll = s.coll.copy()
    if m == 3:
        # propellant: minojme vetem sa duhet per te rimbushur (kursen kohe)
        tm_c = min(full, max(0.0, TMINE * (P["refuel_to"] - fuel)))
        fuel = min(1.0, fuel + tm_c / TMINE)
    else:
        tm_c = full
        coll[m] += tm_c / TMINE
    ch = S()
    ch.ids = s.ids + [c]
    ch.ta = s.ta + [t_arr]
    ch.tm = s.tm[:-1] + [tm, tm_c]
    ch.fuel, ch.coll, ch.vis, ch.parent, ch.look, ch.look2 = fuel, coll, s.vis | {c}, None, None, None
    ch.key = (
        potential(coll, P, t_arr)
        - P["lt"] * min(t_arr + tm_c, WINDOW)
        + P["lf"] * min(fuel, P["burn"] * max(WINDOW - t_arr - tm_c, 0.0))
    )
    return ch


# ================================================================== zgjerimi
def expand_grid(s, cur, base, P):
    stage1 = []
    for c in candidates(cur, s, P):
        b = best_leg(s, cur, c, base, P, P["wgrid"], P["tofgrid"])
        if b is None:
            continue
        _, tm, t_arr, dv = b
        stage1.append((make_child(s, c, tm, t_arr, dv, P), c, tm, t_arr, dv))
    stage1.sort(key=lambda r: r[0].key, reverse=True)
    out = []
    for i, (ch, c, tm, t_arr, dv) in enumerate(stage1):
        if i < REFINE_R:
            w0, tof0 = tm - base, t_arr - (s.ta[-1] + tm)
            ws = sorted({max(0.0, w0 - 6.0), w0, w0 + 6.0})
            tofs = sorted({max(0.5, tof0 * 0.7), tof0, tof0 * 1.3})
            b = best_leg(s, cur, c, base, P, ws, tofs)
            if b is not None:
                ch2 = make_child(s, c, b[1], b[2], b[3], P)
                if ch2.key > ch.key:
                    ch = ch2
        out.append(ch)
    return out


def expand_phase(s, cur, base, P):
    t_dep0 = s.ta[-1] + min(60.0, base)
    est = pos_proxy(cur, t_dep0)
    if P["pw2"] > 0:
        est = np.minimum(est, pos_proxy(cur, t_dep0 + P["pw2"]))
    est[list(s.vis)] = np.inf
    thr = np.where(MATS == 3, P["fmin"], np.where(MATS == 1, P["mmin1"], P["mmin"]))
    est[MASSES < thr] = np.inf
    fin = np.flatnonzero(np.isfinite(est))
    if len(fin) == 0:
        return []
    sel = set(_take(est, fin, P["screen"]))
    for mm in range(4):
        pm = POOLS[mm]
        sel.update(_take(est, pm[np.isfinite(est[pm])], P["screen_pm"]))
    pn = POOLS[int(np.argmin(s.coll))]
    sel.update(_take(est, pn[np.isfinite(est[pn])], P["screen_pm"] * 2))
    if s.fuel < 0.5:
        pf = POOLS[3]
        sel.update(_take(est, pf[np.isfinite(est[pf])], P["screen_pm"] * 2))
    sub = np.fromiter(sel, dtype=int)
    m = len(sub)
    ms, mats = MASSES[sub], MATS[sub]
    full = np.minimum(60.0, TMINE * ms)
    isfuel = mats == 3
    C = np.tile(s.coll, (m, 1))
    rows = np.flatnonzero(~isfuel)
    C[rows, mats[rows]] += full[rows] / TMINE
    cmin, cmean = C.min(1), C.mean(1)

    t_last = s.ta[-1]
    combos = [(w, tof) for w in P["wscreen"] for tof in P["tofscreen"] if t_last + min(60.0, base + w) + tof <= WINDOW]
    if not combos:
        return []
    nc = len(combos)
    tm_arr = np.array([min(60.0, base + w) for w, _ in combos])
    tof_arr = np.array([tof for _, tof in combos])
    ta_arr = t_last + tm_arr + tof_arr
    cache = {}
    R1, V1, R2, V2 = [], [], [], []
    for ci in range(nc):
        td = t_last + tm_arr[ci]
        if td not in cache:
            cache[td] = kepler_state([cur], td)
        r1, v1 = cache[td]
        r2, v2 = kepler_state(sub, ta_arr[ci])
        R1.append(np.broadcast_to(r1, (m, 3)))
        V1.append(np.broadcast_to(v1, (m, 3)))
        R2.append(r2)
        V2.append(v2)
    dv = lambert_dv(np.concatenate(R1), np.concatenate(V1), np.concatenate(R2), np.concatenate(V2), np.repeat(tof_arr, m), iters=26).reshape(nc, m)
    fa = s.fuel - dv / DVF
    isf = isfuel[None, :]
    ok = np.isfinite(dv) & (fa >= P["eps"]) & (isf | (fa >= P["reserve"]))
    tm_c = np.where(isf, np.minimum(full[None, :], np.maximum(0.0, TMINE * (P["refuel_to"] - fa))), full[None, :])
    fnew = np.where(isf, np.minimum(1.0, fa + tm_c / TMINE), fa)
    fval = np.minimum(fnew, P["burn"] * np.maximum(WINDOW - ta_arr[:, None] - tm_c, 0.0))
    a0 = P["alpha"]
    aa = (a0 + (1.0 - a0) * (np.clip(ta_arr, 0.0, WINDOW) / WINDOW) ** P["alpha_pow"])[:, None]
    key = aa * cmin[None, :] + (1.0 - aa) * cmean[None, :] - P["lt"] * np.minimum(ta_arr[:, None] + tm_c, WINDOW) + P["lf"] * fval
    key = np.where(ok, key, -np.inf)
    bi = key.argmax(0)
    best_key = key[bi, np.arange(m)]
    btm = tm_arr[bi]
    btof = tof_arr[bi]

    order = [int(j) for j in np.argsort(-best_key)[: P["topm"]] if np.isfinite(best_key[j])]
    if not order:
        return []
    oj = np.array(order)
    cs = sub[oj]
    m2 = len(cs)
    w0 = btm[oj] - base
    tof0 = btof[oj]
    isf_o = isfuel[oj]
    dws = np.array([dw for dw in P["refw"] for _ in P["reftof"]])
    fts = np.array([ft for _ in P["refw"] for ft in P["reftof"]])
    nc2 = len(dws)
    w = np.maximum(0.0, w0[None, :] + dws[:, None])
    tm2 = np.minimum(60.0, base + w)
    td = t_last + tm2
    tf = np.maximum(0.5, tof0[None, :] * fts[:, None])
    ta2 = td + tf
    curs = np.full(nc2 * m2, cur)
    r1, v1 = kepler_state(curs, td.ravel())
    r2, v2 = kepler_state(np.tile(cs, nc2), ta2.ravel())
    dv2 = lambert_dv(r1, v1, r2, v2, tf.ravel(), iters=32).reshape(nc2, m2)
    fa2 = s.fuel - dv2 / DVF
    ok2 = (ta2 <= WINDOW) & np.isfinite(dv2) & (fa2 >= P["eps"]) & (isf_o[None, :] | (fa2 >= P["reserve"]))
    J = (tm2 - base + tf) + P["lam"] * dv2 / DVF
    J = np.where(ok2, J, np.inf)
    bk = J.argmin(0)
    cols = np.arange(m2)
    bJ = J[bk, cols]
    out = []
    for k in range(m2):
        if np.isfinite(bJ[k]):
            out.append(make_child(s, int(cs[k]), float(tm2[bk[k], k]), float(ta2[bk[k], k]), float(dv2[bk[k], k]), P))
    return out


def finale(s, P):
    """Nje hop i fundit drejt asteroidit qe e rrit me shume objektivin (min). Kthen fëmijen ose None."""
    cur, base = s.ids[-1], s.tm[-1]
    if s.ta[-1] + min(60.0, base) + 0.5 > WINDOW:
        return None
    est = pos_proxy(cur, s.ta[-1] + min(60.0, base))
    est[list(s.vis)] = np.inf
    full = np.minimum(60.0, TMINE * MASSES) / TMINE
    cmin = s.coll.min()
    gain = np.zeros(N)
    for m in range(3):
        others = np.delete(s.coll, m).min()
        sel = MATS == m
        gain[sel] = np.minimum(others, s.coll[m] + full[sel]) - cmin
    gain[~np.isfinite(est)] = 0.0
    cand = np.flatnonzero(gain > 1e-9)
    if len(cand) == 0:
        return None
    sub = cand[np.argsort(est[cand])[:250]]
    g = gain[sub]
    best_sc = np.full(len(sub), -np.inf)
    btm, btof = np.zeros(len(sub)), np.zeros(len(sub))
    t_last = s.ta[-1]
    for w in (0.0, 5.0, 15.0):
        tm = min(60.0, base + w)
        t_dep = t_last + tm
        r1, v1 = kepler_state([cur], t_dep)
        for tof in P["tofscreen"]:
            t_arr = t_dep + tof
            if t_arr > WINDOW:
                continue
            r2, v2 = kepler_state(sub, t_arr)
            dv = lambert_dv(r1[0], v1[0], r2, v2, tof)
            ok = np.isfinite(dv) & (s.fuel - dv / DVF >= P["eps"])
            sc = g - P["lf"] * dv / DVF * 0.1 - 1e-4 * t_arr  # gain dominon; DV/koha vetem si ndarje
            upd = ok & (sc > best_sc)
            best_sc = np.where(upd, sc, best_sc)
            btm = np.where(upd, tm, btm)
            btof = np.where(upd, tof, btof)
    best = None
    P2 = dict(P, reserve=P["eps"])
    for j in np.argsort(-best_sc)[:6]:
        if not np.isfinite(best_sc[j]):
            break
        c = int(sub[j])
        w0, tof0 = btm[j] - base, btof[j]
        b = best_leg(s, cur, c, base, P2, sorted({w0, w0 + 4.0}), sorted({tof0, max(0.5, tof0 * 0.85), tof0 * 1.2}))
        if b is None:
            continue
        ch = make_child(s, c, b[1], b[2], b[3], P2)
        if ch.coll.min() > cmin + 1e-9 and (best is None or ch.coll.min() > best.coll.min()):
            best = ch
    if best is not None:
        best.parent = s
    return best


def add_finales(states, P, max_chain=4):
    """Kthen states + zgjatjet e tyre me hope finale (deri max_chain radhazi)."""
    out = list(states)
    for s in states:
        cur = s
        for _ in range(max_chain):
            ch = finale(cur, P)
            if ch is None:
                break
            out.append(ch)
            cur = ch
    return out


def expand(task):
    s, P = task
    cur = s.ids[-1]
    base = s.tm[-1]
    if PHASE_OK and P["screen"] > 0:
        return expand_phase(s, cur, base, P)
    return expand_grid(s, cur, base, P)


# ================================================================== beam me kosha kohore
def beam_search(pool, inits, B, P, BW, verbose=True, n_elite=20):
    buckets = defaultdict(list)
    bk = (lambda s: int((s.ta[-1] + s.tm[-1]) // BW)) if P.get("bready", 0) else (lambda s: int(s.ta[-1] // BW))
    for s0 in inits:
        buckets[bk(s0)].append(s0)
    elite, cnt = [], itertools.count()
    nb = 0
    while buckets:
        b = min(buckets)
        states = buckets.pop(b)
        states.sort(key=lambda s: s.key, reverse=True)
        if P.get("kappa", 0.0) > 0 or P.get("gamma", 0.0) > 0:
            states = states[: 3 * B]
            for s in states:
                if s.look2 is None:
                    s.look2 = lookahead(s.ids[-1], s.coll, s.vis, s.fuel) if P["gamma"] > 0 else 0.0
                if s.look is None:
                    s.look = near_cost(s) if P["kappa"] > 0 else 0.0
            states.sort(key=lambda s: s.key - (P["gamma"] * s.look2 + P["kappa"] * s.look) / DVF, reverse=True)
        seen, keep = set(), []
        for s in states:
            sig = (s.ids[-1], int(s.ta[-1] // 5))
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(s)
            if len(keep) >= B:
                break
        for s in keep:
            item = (s.coll.min(), next(cnt), s)
            if len(elite) < n_elite:
                heapq.heappush(elite, item)
            elif item[0] > elite[0][0]:
                heapq.heapreplace(elite, item)
        tasks = [(s, P) for s in keep]
        results = pool.map(expand, tasks, chunksize=1) if pool else [expand(t) for t in tasks]
        for s, children in zip(keep, results):
            for ch in children:
                ch.parent = s
                buckets[bk(ch)].append(ch)
        nb += 1
        if verbose and nb % 20 == 0:
            top = keep[0]
            print(
                f"  kosha {b:4d} | t={top.ta[-1]:7.1f} | fuel={top.fuel:.2f} | "
                f"mat={top.coll.round(2)} | elite={max(e[0] for e in elite):.3f}"
            )
    return [e[2] for e in sorted(elite, key=lambda e: -e[0])]


# ================================================================== vleresim
def to_x(s):
    return U.convert_to_chromosome(s.ta + s.tm + s.ids)


def evaluate(states):
    """(f0, s, x) per gjendjet e vlefshme (fitness + kufijt e bounds), me e mira e para."""
    res = []
    for s in states:
        if max(s.tm) > 60.0 + 1e-9 or max(s.ta) > WINDOW or min(s.tm) < 0.0:
            continue
        try:
            x = to_x(s)
            f = U.udp.fitness(x)
        except Exception:
            continue
        if f[1] == 0 and max(f[2:]) <= 0:
            res.append((f[0], s, x))
    res.sort(key=lambda r: r[0])
    return res


def chain_of(s):
    out = []
    while s is not None:
        out.append(s)
        s = s.parent
    return out[::-1]


def save(x, path):
    with open(path, "w") as fh:
        json.dump(
            {"decisionVector": [float(v) for v in x], "problem": "mine-the-belt", "challenge": "spoc-mining"},
            fh,
        )


def parse_floats(txt):
    return tuple(float(v) for v in txt.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--start-id", type=int, default=None)
    ap.add_argument("--stratified", action="store_true", help="startet neper materialet 0,1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--bw", type=float, default=12.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-phase", action="store_true", help="fik screening-un fazor (grid v4)")
    for k, v in DEFAULT_P.items():
        if not isinstance(v, tuple):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--tof-screen", default=",".join(f"{v:g}" for v in DEFAULT_P["tofscreen"]))
    ap.add_argument("--w-screen", default=",".join(f"{v:g}" for v in DEFAULT_P["wscreen"]))
    ap.add_argument("--out", default="solution_beam_v12.json")
    args = ap.parse_args()

    P0 = dict(DEFAULT_P)
    for k, v in DEFAULT_P.items():
        if not isinstance(v, tuple):
            P0[k] = getattr(args, k)
    P0["tofscreen"] = parse_floats(args.tof_screen)
    P0["wscreen"] = parse_floats(args.w_screen)
    if args.no_phase:
        P0["screen"] = 0
    print(f"Screening fazor aktiv: {PHASE_OK and P0['screen'] > 0}  (validimi me pykep: {PHASE_OK})")

    rng = np.random.default_rng(args.seed)
    pool = Pool(args.jobs) if args.jobs > 1 else None
    best_x, best_f = None, 1e9

    if args.start_id is not None:
        starts = [args.start_id]
    elif args.stratified:
        starts = [int(rng.choice(POOLS[r % 3])) for r in range(args.starts)]
    else:
        starts = [int(rng.integers(N)) for _ in range(args.starts)]

    for r, start in enumerate(starts):
        t = time.time()
        print(f"Start {r}: asteroidi {start} (materiali {MATS[start]}), beam={args.beam}")
        elite = beam_search(pool, [root(start)], args.beam, P0, args.bw)
        ev = evaluate(add_finales(elite, P0) if PHASE_OK else elite)
        if not ev:
            print("  -> asnje elite e vlefshme")
            continue
        cur_f, cur_s, cur_x = ev[0]
        print(f"  -> raundi 0: objektivi = {cur_f:.4f} | {time.time() - t:.0f}s")

        for rd in range(1, args.rounds + 1):
            ch = chain_of(cur_s)
            frac = rng.uniform(0.3, 0.9)
            anc = ch[max(0, int(len(ch) * frac) - 1)]
            P = dict(P0)
            P["lt"] = P0["lt"] * rng.uniform(0.6, 1.5)
            P["lf"] = P0["lf"] * rng.uniform(0.6, 1.5)
            P["gamma"] = P0["gamma"] * rng.uniform(0.5, 1.8)
            P["alpha"] = float(np.clip(P0["alpha"] + rng.uniform(-0.15, 0.15), 0.0, 0.8))
            el2 = beam_search(pool, [anc], args.beam * 2, P, args.bw, verbose=False)
            ev2 = evaluate(add_finales(el2, P) if PHASE_OK else el2)
            if ev2 and ev2[0][0] < cur_f:
                cur_f, cur_s, cur_x = ev2[0]
                print(f"  -> raundi {rd}: permiresim {cur_f:.4f} (cut={frac:.2f})")
            else:
                print(f"  -> raundi {rd}: pa permiresim")

        if cur_f < best_f:
            best_x, best_f = cur_x, cur_f
            save(best_x, args.out)
            print(f"  ** me i miri deri tani: {best_f:.4f} (ruajtur ne {args.out})")

    if pool:
        pool.close()
    if best_x is None:
        print("Asnje zgjidhje e vlefshme.")
        return
    print(f"\nMe e miri: {best_f:.4f}  (lider: -10.687)")
    U.udp.pretty(best_x)


if __name__ == "__main__":
    main()
