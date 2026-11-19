"""
beam_search.py - Zgjidhje me Beam Search per SpOC "Mine the Belt" (ESA).

PROBLEMI
    10000 asteroide rreth nje ylli. Anija fillon ne nje asteroid dhe fluturon asteroid pas asteroidi
    (transfer Lambert; kostoja = DV = karburant). Ne cdo asteroid minon nje nga 3 materialet (0, 1, 2)
    ose karburant (3). Minimi ka shpejtesi 1/30 mase per dite, masa e asteroidit <= 1, cdo asteroid vizitohet vetem nje here.
    Misioni zgjat 1827 dite. Objektivi = -min(masa e materialit 0, 1, 2); sa me negativ, aq me mire.

IDEJA
    Beam search me "kosha kohore": gjendjet (zinxhiret e pjesshme) grupohen sipas kohes kur jane gati
    per nisje, dhe ne cdo kose mbahen vetem B gjendjet me te mira. Per cdo gjendje hopet gjenerohen ne 3 faza:
      1. candidates: pos_proxy (vleresim shume i shpejte i DV nga distanca e pozicioneve) -> ~200 asteroide
      2. screen:     Lambert i vektorizuar per shume (pritje, tof) njekohesisht -> 30 kandidatet me "key" me te madh
      3. refine:     kerkim me i imet i (pritje, tof) per ata 30 -> gjendjet femije
    "key" i nje gjendjeje = peshe * min(materialet) + (1 - peshe) * mesatarja(materialeve)
                            - LT * koha + LF * vlera e karburantit qe ka mbetur
    (pesha e minit rritet nga ALPHA drejt 1 gjate misionit). Pas beam-it, finale() shton deri 4 hope te fundit
    per te rritur minin, dhe cdo zgjidhje verifikohet me udp.fitness (pykep).
    RAUNDET: merr nje paraardhes te rastesishem te zinxhirit me te mire dhe e ben beam-in prape nga aty
    (beam me i gjere, parametra pak te ndryshuar); pranohet vetem nese objektivi permireson.

PERDORIMI
    python beam_search.py --beam 80 --stratified --seed 1 --jobs 4 --out best.json
    python beam_search.py --resume best.json --rbeam 100 --cut-lo 0.75 --cut-hi 0.97 --rounds 40 --out best.json
    python make_submission.py --in best.json --out submission_upload.json
Skedari --out ndryshohet VETEM kur gjendet zgjidhje me e mire se ajo qe ka, ndaj disa dritare mund te shkruajne te njejtin skedar.
"""
import argparse
import heapq
import itertools
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from multiprocessing import Pool

import numpy as np
import pykep as pk
import udp as U

# ============================================================ konstantet dhe te dhenat
MU = U.MU_TRAPPIST
T0 = U.T_START.mjd2000
WINDOW = U.udp.mission_window[1]  # 1827 dite
DVF = U.DV_per_fuel  # 1 njesi karburanti = 10000 m/s
TMINE = U.TIME_TO_MINE_FULLY  # 30 dite per 1 njesi mase
N = U.udp.n
TWO_PI = 2.0 * np.pi

A, ECC, INC, OM, ARG, M0 = (U.data[:, i] for i in range(1, 7))  # elementet orbitale
MASSES, MATS = U.asteroid_masses, U.asteroid_materials
VC = np.sqrt(MU / A)  # shpejtesia rrethore
NMEAN = np.sqrt(MU / A**3)  # levizja mesatare (rad/s)
POOLS = {m: np.where(MATS == m)[0] for m in range(4)}  # asteroidet sipas materialit

# Parametrat (te gjithe skalaret behen opsione ne rreshtin e komandes: --lt, --lam, ...)
DEFAULT_P = dict(
    lt=0.004,  # vlera e nje dite (kosto e kohes ne "key")
    lf=0.3,  # vlera e nje njesie karburanti ne "key"
    lam=40.0,  # dite ekuivalente per 1 njesi karburanti (zgjedh (pritje, tof) ne rafinim)
    alpha=0.1,  # pesha fillestare e minit (rritet drejt 1)
    alpha_pow=3.0,  # sa vone rritet pesha e minit
    gamma=0.5,  # pesha e lookahead-it (sa te lire jane hopet e ardhshme)
    mmin=0.30,  # masa minimale e asteroideve te materialit 0 dhe 2
    mmin1=0.15,  # ... e materialit 1 (te rralle)
    fmin=0.40,  # ... e asteroideve te karburantit
    reserve=0.15,  # karburant minimal pas nje hopi drejt materiali
    refuel_to=1.0,  # deri ku mbushim karburantin ne nje ndalese karburanti
    burn=0.006,  # konsum i pritur karburanti (njesi/dite) per vleresimin ne fund
    eps=1e-4,  # margjine sigurie e karburantit pas cdo hopi
    screen=200,  # sa asteroide te afert (sipas pos_proxy) hyjne ne screening
    screen_pm=25,  # kandidate shtese per cdo material
    topm=30,  # sa kandidate kalojne ne rafinim
    pw2=10.0,  # pos_proxy llogaritet edhe per nisje te vonuar me kaq dite
    wscreen=(0.0, 10.0),  # pritjet e provuara ne screening (dite)
    tofscreen=(3.0, 6.0, 10.0, 16.0, 24.0, 36.0, 52.0),  # kohet e fluturimit ne screening
    refw=(-5.0, 0.0, 5.0),  # ndryshimet e pritjes ne rafinim
    reftof=(0.75, 0.9, 1.0, 1.15, 1.35),  # shumezuesit e tof ne rafinim
)


# ============================================================ fizika
def transfer_dv(a_from, t_dep, a_to, t_arr):
    """DV i sakte (m/s) me pykep per nje hop; None nese deshton. Koha ne dite nga fillimi i misionit."""
    try:
        r1, v1 = U.asteroids[a_from].eph(T0 + t_dep)
        r2, v2 = U.asteroids[a_to].eph(T0 + t_arr)
        lam = pk.lambert_problem(r1=r1, r2=r2, tof=(t_arr - t_dep) * pk.DAY2SEC, mu=MU, cw=False, max_revs=0)
        return np.linalg.norm(np.array(v1) - np.array(lam.get_v1()[0])) + np.linalg.norm(
            np.array(v2) - np.array(lam.get_v2()[0])
        )
    except Exception:
        return None


def kepler_state(idx, t):
    """Pozicioni dhe shpejtesia (r, v), shape (m,3), e asteroideve idx ne kohen t (dite). Vektorizuar."""
    idx = np.asarray(idx)
    a, e = A[idx], ECC[idx]
    M = np.mod(M0[idx] + NMEAN[idx] * (t * 86400.0), TWO_PI)
    E = M + e * np.sin(M)
    for _ in range(8):  # ekuacioni i Keplerit me Newton
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

    def rot(px, py):  # nga plani orbital ne hapesiren 3D
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
    """DV total (m/s) i transferit Lambert prograde pa rrotullime te plota; inf nese s'ka zgjidhje.
    Variabla universale (Curtis) me bisektim, e vektorizuar: r1,v1 (3,) ose (m,3); tau_days skalar ose (m,)."""
    with np.errstate(all="ignore"):
        r1, v1 = np.atleast_2d(r1), np.atleast_2d(v1)
        tau = np.asarray(tau_days, dtype=float) * 86400.0
        m = len(r2)
        r1n, r2n = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
        cosd = np.einsum("ij,ij->i", r2, np.broadcast_to(r1, (m, 3))) / (r1n * r2n)
        crossz = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
        sind = np.sqrt(np.clip(1.0 - cosd**2, 0.0, None)) * np.where(crossz >= 0, 1.0, -1.0)
        Ac = sind * np.sqrt(r1n * r2n / np.maximum(1.0 - cosd, 1e-12))
        sm = np.sqrt(MU)

        def F(z):  # F(z) = 0 <=> koha e fluturimit perputhet me tau
            C, S = _stumpff(z)
            y = r1n + r2n + Ac * (z * S - 1.0) / np.sqrt(C)
            yy = np.where(y > 0, y, 1.0)
            val = (yy / C) ** 1.5 * S + Ac * np.sqrt(yy) - sm * tau
            return np.where(y > 0, val, -1.0), y

        lo, hi = np.full(m, -60.0), np.full(m, TWO_PI**2 - 1e-3)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            neg = F(mid)[0] < 0
            lo, hi = np.where(neg, mid, lo), np.where(neg, hi, mid)
        f, y = F(0.5 * (lo + hi))
        valid = (y > 0) & (np.abs(f) < 1e-3 * sm * tau) & (np.abs(Ac) > 1e-9)
        g = Ac * np.sqrt(y / MU)
        vl1 = (r2 - (1.0 - y / r1n)[:, None] * r1) / g[:, None]
        vl2 = ((1.0 - y / r2n)[:, None] * r2 - r1) / g[:, None]
        dv = np.linalg.norm(v1 - vl1, axis=1) + np.linalg.norm(v2 - vl2, axis=1)
        return np.where(valid & np.isfinite(dv), dv, np.inf)


def check_physics():
    """Kontroll ne fillim: propagimi dhe Lambert-i i vektorizuar duhet te perputhen me pykep."""
    ids = np.random.default_rng(123).integers(N, size=12)
    for i in ids[:4]:
        r, v = kepler_state([i], 137.0)
        r0, v0 = U.asteroids[int(i)].eph(T0 + 137.0)
        assert np.linalg.norm(r[0] - np.array(r0)) < 1e-5 * np.linalg.norm(r0), "pozicioni nuk perputhet me pykep"
        assert np.linalg.norm(v[0] - np.array(v0)) < 1e-5 * np.linalg.norm(v0), "shpejtesia nuk perputhet me pykep"
    cur, tgt = int(ids[0]), ids[1:]
    r1, v1 = kepler_state([cur], 50.0)
    r2, v2 = kepler_state(tgt, 70.0)
    dvv = lambert_dv(r1[0], v1[0], r2, v2, 20.0)
    errs = []
    for k, c in enumerate(tgt):
        ex = transfer_dv(cur, 50.0, int(c), 70.0)
        if ex is not None and np.isfinite(dvv[k]):
            errs.append(abs(ex - dvv[k]) / max(ex, 1.0))
    assert len(errs) >= 4 and np.median(errs) < 1e-3, "Lambert-i i vektorizuar nuk perputhet me pykep"


# ============================================================ zgjedhja e kandidateve
GRID_DAYS = 2000
PTOFS = (4.0, 8.0, 12.0, 18.0, 27.0)
_G = None


def grid():
    """Pozicionet/shpejtesite e te 10000 asteroideve per cdo dite (float32, ~480 MB). Ndertohet nje here per proces."""
    global _G
    if _G is None:
        gp = np.empty((GRID_DAYS, N, 3), dtype=np.float32)
        gv = np.empty((GRID_DAYS, N, 3), dtype=np.float32)
        idx = np.arange(N)
        for k in range(GRID_DAYS):
            gp[k], gv[k] = kepler_state(idx, float(k))
        _G = (gp, gv)
    return _G


def grid_pv(t):
    """Pozicioni (interpolim Hermite) dhe shpejtesia (linear) e te gjithe asteroideve ne kohen t, nga grid-i."""
    gp, gv = grid()
    t = min(max(t, 0.0), GRID_DAYS - 1.001)
    i = int(t)
    f = t - i
    h = 86400.0
    h00, h10 = 2 * f**3 - 3 * f**2 + 1, f**3 - 2 * f**2 + f
    h01, h11 = -2 * f**3 + 3 * f**2, f**3 - f**2
    p = h00 * gp[i] + (h10 * h) * gv[i] + h01 * gp[i + 1] + (h11 * h) * gv[i + 1]
    return p, (1.0 - f) * gv[i] + f * gv[i + 1]


def pos_proxy(cur, t_dep):
    """Vleresim i shpejte i DV (m/s) nga asteroidi cur (nisje ne t_dep) drejt te gjithe te tjereve:
    distanca e pozicioneve ne t_dep+tof pjestuar me tof, e kombinuar me shpejtesine relative; minimumi mbi disa tof."""
    best = np.full(N, np.inf)
    for tf in PTOFS:
        rc, vc = kepler_state([cur], t_dep + tf)
        rt, vt = grid_pv(t_dep + tf)
        d = np.linalg.norm(rt - rc, axis=1) / (tf * 86400.0)
        best = np.minimum(best, np.hypot(d, 0.5 * np.linalg.norm(vt - vc, axis=1)))
    return best


def est_dv(cur, idx):
    """Vleresim i trashe i DV (m/s) nga cur drejt asteroideve idx: diferenca e shpejtesive rrethore + ndryshimi i planit."""
    dva = np.abs(VC[cur] - VC[idx])
    c = np.cos(INC[cur]) * np.cos(INC[idx]) + np.sin(INC[cur]) * np.sin(INC[idx]) * np.cos(OM[cur] - OM[idx])
    th = np.arccos(np.clip(c, -1.0, 1.0))
    return np.hypot(dva, 0.5 * (VC[cur] + VC[idx]) * th)


def lookahead(s):
    """Sa te lira jane hopet e ardhshme: DV mesatar i 3 asteroideve me te lira te materialit qe na duhet (ose karburant)."""
    pool = POOLS[3] if s.fuel < 0.3 else POOLS[int(np.argmin(s.coll))]
    e = est_dv(s.ids[-1], pool)
    e[np.isin(pool, np.fromiter(s.vis, dtype=int, count=len(s.vis)))] = np.inf
    k = min(3, len(e))
    near = np.partition(e, k - 1)[:k]
    near = near[np.isfinite(near)]
    return float(near.mean()) if len(near) else float(VC.mean())


def _take(est, idx, k):
    """k indekset me est me te vogel brenda idx."""
    k = min(k, len(idx))
    if k <= 0:
        return []
    if k == len(idx):
        return idx.tolist()
    return idx[np.argpartition(est[idx], k - 1)[:k]].tolist()


# ============================================================ gjendja e beam-it
class S:
    """Nje zinxhir i pjesshem: asteroidet, kohet e mberritjes (ta), kohet e minimit (tm), karburanti, masat e mbledhura."""

    __slots__ = ("ids", "ta", "tm", "fuel", "coll", "vis", "key", "parent", "look")

    def __getstate__(self):  # per multiprocessing: pa prindin (qe te mos dergohet gjithe zinxhiri)
        return (self.ids, self.ta, self.tm, self.fuel, self.coll, self.vis, self.key)

    def __setstate__(self, st):
        self.ids, self.ta, self.tm, self.fuel, self.coll, self.vis, self.key = st
        self.parent = self.look = None


def root(start):
    s = S()
    s.ids, s.ta, s.tm, s.fuel, s.coll, s.vis, s.key, s.parent, s.look = [start], [0.0], [0.0], 1.0, np.zeros(3), {start}, 0.0, None, None
    return s


def state_key(cmin, cmean, t, tm_c, fuel, P):
    """Vlera e nje gjendjeje (sa me e madhe aq me mire). cmin, cmean: min dhe mesatarja e 3 materialeve; t: koha e
    mberritjes; tm_c: koha e minimit ketu; fuel: karburanti pas minimit. Pranon numra ose vargje numpy."""
    a = P["alpha"] + (1.0 - P["alpha"]) * (np.clip(t, 0.0, WINDOW) / WINDOW) ** P["alpha_pow"]
    fuel_value = np.minimum(fuel, P["burn"] * np.maximum(WINDOW - t - tm_c, 0.0))  # karburanti i pa perdorur ne fund eshte pa vlere
    return a * cmin + (1.0 - a) * cmean - P["lt"] * np.minimum(t + tm_c, WINDOW) + P["lf"] * fuel_value


def make_child(s, c, tm, t_arr, dv, P):
    """Gjendja pas nje hopi nga s drejt asteroidit c. tm = koha e minimit ne asteroidin e fundit te s (mining + pritje)."""
    fuel = s.fuel - dv / DVF
    full = min(60.0, TMINE * MASSES[c])  # koha per te minuar krejt asteroidin
    coll = s.coll.copy()
    if MATS[c] == 3:  # karburant: minojme vetem sa duhet per ta mbushur
        tm_c = min(full, max(0.0, TMINE * (P["refuel_to"] - fuel)))
        fuel = min(1.0, fuel + tm_c / TMINE)
    else:
        tm_c = full
        coll[MATS[c]] += tm_c / TMINE
    ch = S()
    ch.ids, ch.ta, ch.tm = s.ids + [c], s.ta + [t_arr], s.tm[:-1] + [tm, tm_c]
    ch.fuel, ch.coll, ch.vis, ch.parent, ch.look = fuel, coll, s.vis | {c}, None, None
    ch.key = state_key(coll.min(), coll.mean(), t_arr, tm_c, fuel, P)
    return ch


# ============================================================ zgjerimi i nje gjendjeje (3 fazat)
def candidates(s, cur, base, P):
    """Faza 1: asteroidet e pavizituar me mase te mjaftueshme qe jane me afer sipas pos_proxy (+ per cdo material)."""
    t_dep0 = s.ta[-1] + min(60.0, base)
    est = pos_proxy(cur, t_dep0)
    if P["pw2"] > 0:
        est = np.minimum(est, pos_proxy(cur, t_dep0 + P["pw2"]))
    est[list(s.vis)] = np.inf
    est[MASSES < np.where(MATS == 3, P["fmin"], np.where(MATS == 1, P["mmin1"], P["mmin"]))] = np.inf
    fin = np.flatnonzero(np.isfinite(est))
    if len(fin) == 0:
        return None
    sel = set(_take(est, fin, P["screen"]))
    for m in range(4):
        sel.update(_take(est, POOLS[m][np.isfinite(est[POOLS[m]])], P["screen_pm"]))
    need = POOLS[int(np.argmin(s.coll))]  # materiali me te paket i mbledhur
    sel.update(_take(est, need[np.isfinite(est[need])], P["screen_pm"] * 2))
    if s.fuel < 0.5:
        fuel_pool = POOLS[3]
        sel.update(_take(est, fuel_pool[np.isfinite(est[fuel_pool])], P["screen_pm"] * 2))
    return np.fromiter(sel, dtype=int)


def screen(s, cur, base, sub, P):
    """Faza 2: per cdo kandidat provon (pritje, tof) te trashe me nje thirrje Lambert dhe ruan kombinimin me "key" me te madh.
    Kthen (kandidatet me te mire, pritja, tof) per topm kandidatet, ose None."""
    m = len(sub)
    mats = MATS[sub]
    full = np.minimum(60.0, TMINE * MASSES[sub])
    isf = (mats == 3)[None, :]
    C = np.tile(s.coll, (m, 1))  # masat e mbledhura nese vizitojme secilin kandidat
    rows = np.flatnonzero(mats != 3)
    C[rows, mats[rows]] += full[rows] / TMINE
    cmin, cmean = C.min(1), C.mean(1)

    t_last = s.ta[-1]
    combos = [(w, tof) for w in P["wscreen"] for tof in P["tofscreen"] if t_last + min(60.0, base + w) + tof <= WINDOW]
    if not combos:
        return None
    nc = len(combos)
    tm_arr = np.array([min(60.0, base + w) for w, _ in combos])
    tof_arr = np.array([tof for _, tof in combos])
    ta_arr = t_last + tm_arr + tof_arr
    cache, R1, V1, R2, V2 = {}, [], [], [], []
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
    dv = lambert_dv(np.concatenate(R1), np.concatenate(V1), np.concatenate(R2), np.concatenate(V2), np.repeat(tof_arr, m)).reshape(nc, m)
    fa = s.fuel - dv / DVF  # karburanti pas hopit
    ok = np.isfinite(dv) & (fa >= P["eps"]) & (isf | (fa >= P["reserve"]))
    tm_c = np.where(isf, np.minimum(full[None, :], np.maximum(0.0, TMINE * (P["refuel_to"] - fa))), full[None, :])
    fnew = np.where(isf, np.minimum(1.0, fa + tm_c / TMINE), fa)
    key = state_key(cmin[None, :], cmean[None, :], ta_arr[:, None], tm_c, fnew, P)
    key = np.where(ok, key, -np.inf)
    bi = key.argmax(0)  # kombinimi me i mire per cdo kandidat
    best_key = key[bi, np.arange(m)]
    oj = np.array([int(j) for j in np.argsort(-best_key)[: P["topm"]] if np.isfinite(best_key[j])], dtype=int)
    if len(oj) == 0:
        return None
    return sub[oj], tm_arr[bi][oj] - base, tof_arr[bi][oj]


def refine(s, cur, base, cs, w0, tof0, P):
    """Faza 3: per cdo kandidat te zgjedhur provon (pritje, tof) rreth zgjidhjes se trashe dhe zgjedh ate me
    J = dite + LAM * karburant me te vogel. Kthen gjendjet femije."""
    m2 = len(cs)
    isf = (MATS[cs] == 3)[None, :]
    dws = np.array([dw for dw in P["refw"] for _ in P["reftof"]])
    fts = np.array([ft for _ in P["refw"] for ft in P["reftof"]])
    nc2 = len(dws)
    t_last = s.ta[-1]
    w = np.maximum(0.0, w0[None, :] + dws[:, None])
    tm2 = np.minimum(60.0, base + w)
    td = t_last + tm2
    tf = np.maximum(0.5, tof0[None, :] * fts[:, None])
    ta2 = td + tf
    r1, v1 = kepler_state(np.full(nc2 * m2, cur), td.ravel())
    r2, v2 = kepler_state(np.tile(cs, nc2), ta2.ravel())
    dv2 = lambert_dv(r1, v1, r2, v2, tf.ravel(), iters=32).reshape(nc2, m2)
    fa2 = s.fuel - dv2 / DVF
    ok = (ta2 <= WINDOW) & np.isfinite(dv2) & (fa2 >= P["eps"]) & (isf | (fa2 >= P["reserve"]))
    J = np.where(ok, (tm2 - base + tf) + P["lam"] * dv2 / DVF, np.inf)
    bk = J.argmin(0)
    cols = np.arange(m2)
    return [
        make_child(s, int(cs[k]), float(tm2[bk[k], k]), float(ta2[bk[k], k]), float(dv2[bk[k], k]), P)
        for k in range(m2)
        if np.isfinite(J[bk[k], k])
    ]


def expand(task):
    """Kthen gjendjet femije te s (te gjitha hopet e mundshme dhe te mira nga asteroidi i fundit)."""
    s, P = task
    cur, base = s.ids[-1], s.tm[-1]
    sub = candidates(s, cur, base, P)
    top = screen(s, cur, base, sub, P) if sub is not None else None
    return refine(s, cur, base, *top, P) if top is not None else []


# ============================================================ fundi i misionit
def best_leg(s, cur, c, base, P, ws, tofs):
    """Me pykep, per hopin cur -> c: (J, tm, t_arr, dv) me J minimal mbi pritjet ws dhe kohet tofs, ose None."""
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


def finale(s, P):
    """Nje hop i fundit drejt asteroidit qe e rrit me shume min-in e materialeve. Kthen femijen ose None."""
    cur, base = s.ids[-1], s.tm[-1]
    if s.ta[-1] + min(60.0, base) + 0.5 > WINDOW:
        return None
    est = pos_proxy(cur, s.ta[-1] + min(60.0, base))
    est[list(s.vis)] = np.inf
    full = np.minimum(60.0, TMINE * MASSES) / TMINE
    cmin = s.coll.min()
    gain = np.zeros(N)  # sa rritet min-i nese vizitojme asteroidin
    for m in range(3):
        others = np.delete(s.coll, m).min()
        gain[MATS == m] = np.minimum(others, s.coll[m] + full[MATS == m]) - cmin
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
            sc = g - P["lf"] * dv / DVF * 0.1 - 1e-4 * t_arr  # rritja e min-it dominon; DV/koha vetem per ndarje
            upd = ok & (sc > best_sc)
            best_sc = np.where(upd, sc, best_sc)
            btm, btof = np.where(upd, tm, btm), np.where(upd, tof, btof)
    best, P2 = None, dict(P, reserve=P["eps"])
    for j in np.argsort(-best_sc)[:6]:
        if not np.isfinite(best_sc[j]):
            break
        c, w0, tof0 = int(sub[j]), btm[j] - base, btof[j]
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
    """states + zgjatjet e tyre me deri max_chain hope finale radhazi."""
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


# ============================================================ beam search
def beam_search(pool, inits, B, P, BW, verbose=True, n_elite=20):
    """Beam me kosha kohore: kosha = (mberritje + minim) // BW dite; ne cdo kose mbahen B gjendjet me te mira.
    Kthen n_elite gjendjet me min-in me te madh te materialeve."""
    bucket = lambda s: int((s.ta[-1] + s.tm[-1]) // BW)
    buckets = defaultdict(list)
    for s0 in inits:
        buckets[bucket(s0)].append(s0)
    elite, cnt, nb = [], itertools.count(), 0
    while buckets:
        b = min(buckets)  # kosha me e hershme
        states = sorted(buckets.pop(b), key=lambda s: s.key, reverse=True)
        if P["gamma"] > 0:  # korrigjim me lookahead per 3B gjendjet me te mira
            states = states[: 3 * B]
            for s in states:
                if s.look is None:
                    s.look = lookahead(s)
            states.sort(key=lambda s: s.key - P["gamma"] * s.look / DVF, reverse=True)
        seen, keep = set(), []
        for s in states:  # pa dublikate (i njejti asteroid ne te njejten kohe afersisht)
            sig = (s.ids[-1], int(s.ta[-1] // 5))
            if sig not in seen:
                seen.add(sig)
                keep.append(s)
                if len(keep) >= B:
                    break
        for s in keep:  # mbaj elitat (min-i me i madh)
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
                buckets[bucket(ch)].append(ch)
        nb += 1
        if verbose and nb % 20 == 0:
            top = keep[0]
            print(f"  kosha {b:4d} | t={top.ta[-1]:7.1f} | fuel={top.fuel:.2f} | mat={top.coll.round(2)} | elite={max(e[0] for e in elite):.3f}")
    return [e[2] for e in sorted(elite, key=lambda e: -e[0])]


# ============================================================ vleresimi, ruajtja, vazhdimi
def to_x(s):
    return U.convert_to_chromosome(s.ta + s.tm + s.ids)


def evaluate(states):
    """[(objektivi, gjendja, x)] per gjendjet e vlefshme sipas udp.fitness, me e mira e para."""
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
    return sorted(res, key=lambda r: r[0])


def chain_of(s):
    out = []
    while s is not None:
        out.append(s)
        s = s.parent
    return out[::-1]


def save(x, path):
    with open(path, "w") as fh:
        json.dump({"decisionVector": [float(v) for v in x], "problem": "mine-the-belt", "challenge": "spoc-mining"}, fh)


def file_objective(path):
    """Objektivi i zgjidhjes ne skedar nese eshte e vlefshme, perndryshe None."""
    try:
        with open(path) as fh:
            d = json.load(fh)
        f = U.udp.fitness((d[0] if isinstance(d, list) else d)["decisionVector"])
        return float(f[0]) if f[1] == 0 and max(f[2:]) <= 0 else None
    except Exception:
        return None


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def save_if_better(x, f, path):
    """Shkruan x ne path VETEM nese f eshte me i miri se ai qe ka skedari. Kyc me skedar .lock qe disa procese te
    shkruajne te njejtin skedar. True nese u shkrua."""
    lock, t0 = path + ".lock", time.time()
    while True:  # prit derisa asnje proces tjeter te mos po shkruaje
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.time() - t0 > 60:  # kyc i vjeter (proces i mbyllur me force)
                _rm(lock)
                t0 = time.time()
            time.sleep(0.2)
    try:
        old = file_objective(path)
        if old is not None and old <= f + 1e-12:
            return False
        tmp = f"{path}.{os.getpid()}.tmp"
        save(x, tmp)
        for _ in range(8):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # p.sh. OneDrive/antivirusi e mban skedarin per pak
                time.sleep(0.5)
        else:
            save(x, path)
            _rm(tmp)
        return True
    finally:
        _rm(lock)


def rebuild(x, P):
    """Rindertoon gjendjen (me zinxhirin e prinderve) nga nje decisionVector ekzistues, qe te vazhdojme nga ai."""
    x = np.asarray(x, dtype=float)
    ta, tm, ids = x[:N], x[N : 2 * N], x[2 * N :].astype(int)
    K = 1  # numri i asteroideve te vizituar
    while K < N and ta[K] > ta[K - 1] and ta[K] <= WINDOW:
        K += 1
    s = root(int(ids[0]))
    for i in range(1, K):
        dv = transfer_dv(int(ids[i - 1]), float(ta[i - 1] + tm[i - 1]), int(ids[i]), float(ta[i]))
        if dv is None:
            return None
        ch = make_child(s, int(ids[i]), float(tm[i - 1]), float(ta[i]), dv, P)
        ch.parent, s = s, ch
    return s


def run_rounds(pool, args, P0, rng, cur, deadline):
    """Raundet e permiresimit. cur = (objektivi, gjendja, x). Ruan ne --out cdo permiresim menjehere."""
    cur_f, cur_s, cur_x = cur
    rbeam = args.rbeam or args.beam * 2
    for rd in range(1, args.rounds + 1):
        if deadline and time.time() > deadline:
            print("  -> koha e caktuar (--hours) mbaroi")
            break
        ch = chain_of(cur_s)
        frac = rng.uniform(args.cut_lo, args.cut_hi)
        anc = ch[max(0, int(len(ch) * frac) - 1)]  # paraardhesi nga ku nisim prape
        P = dict(P0)  # parametra pak te ndryshuar (diversitet)
        P["lt"] = P0["lt"] * rng.uniform(0.6, 1.5)
        P["lf"] = P0["lf"] * rng.uniform(0.6, 1.5)
        P["gamma"] = P0["gamma"] * rng.uniform(0.5, 1.8)
        P["alpha"] = float(np.clip(P0["alpha"] + rng.uniform(-0.15, 0.15), 0.0, 0.8))
        ev = evaluate(add_finales(beam_search(pool, [anc], rbeam, P, args.bw, verbose=False), P))
        if ev and ev[0][0] < cur_f:
            cur_f, cur_s, cur_x = ev[0]
            saved = save_if_better(cur_x, cur_f, args.out)
            print(f"  -> raundi {rd}: permiresim {cur_f:.4f} (cut={frac:.2f})" + (f" | ruajtur ne {args.out}" if saved else ""))
        else:
            print(f"  -> raundi {rd}: pa permiresim (cut={frac:.2f}, {time.strftime('%H:%M:%S')})")
    return cur_f, cur_s, cur_x


# ============================================================ programi kryesor
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beam", type=int, default=6, help="gjeresia e beam-it ne raundin 0")
    ap.add_argument("--starts", type=int, default=1, help="sa starte te rinj")
    ap.add_argument("--start-id", type=int, default=None, help="asteroidi i startit (ndryshe i rastesishem)")
    ap.add_argument("--stratified", action="store_true", help="startet nga asteroide te materialeve 0,1,2 me radhe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="procese paralele (cdo proces ~0.5 GB RAM)")
    ap.add_argument("--bw", type=float, default=12.0, help="gjeresia e koshes kohore (dite)")
    ap.add_argument("--rounds", type=int, default=3, help="sa raunde permiresimi")
    ap.add_argument("--rbeam", type=int, default=0, help="beam i raundeve (0 = 2*beam)")
    ap.add_argument("--cut-lo", type=float, default=0.3, help="raundet nisin nga nje pike e zinxhirit ne [cut-lo, cut-hi]")
    ap.add_argument("--cut-hi", type=float, default=0.9)
    ap.add_argument("--hours", type=float, default=0.0, help="mos nis raunde te reja pas kaq oresh (0 = pa limit)")
    ap.add_argument("--resume", default=None, help="vazhdo nga nje zgjidhje .json: ben vetem raunde permiresimi")
    ap.add_argument("--out", default="best.json", help="ruhet ketu VETEM nese eshte me e mira se ajo qe ka skedari")
    for k, v in DEFAULT_P.items():
        if not isinstance(v, tuple):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--tof-screen", default=",".join(f"{v:g}" for v in DEFAULT_P["tofscreen"]))
    ap.add_argument("--w-screen", default=",".join(f"{v:g}" for v in DEFAULT_P["wscreen"]))
    args = ap.parse_args()

    P0 = {k: (v if isinstance(v, tuple) else getattr(args, k)) for k, v in DEFAULT_P.items()}
    P0["tofscreen"] = tuple(float(v) for v in args.tof_screen.split(","))
    P0["wscreen"] = tuple(float(v) for v in args.w_screen.split(","))
    check_physics()
    grid()  # ndertoje para proceseve: ne Linux ndahet midis tyre; ne Windows cdo proces e ndertohet vete
    rng = np.random.default_rng(args.seed)
    deadline = time.time() + args.hours * 3600.0 if args.hours > 0 else 0.0
    best = None  # (objektivi, gjendja, x)

    with (Pool(args.jobs) if args.jobs > 1 else nullcontext()) as pool:
        if args.resume:
            with open(args.resume) as fh:
                d = json.load(fh)
            s0 = rebuild((d[0] if isinstance(d, list) else d)["decisionVector"], P0)
            ev = evaluate([s0]) if s0 is not None else []
            if not ev:
                raise SystemExit(f"Nuk u rindertua dot nje zgjidhje e vlefshme nga {args.resume}")
            print(f"Rindertuar nga {args.resume}: objektivi = {ev[0][0]:.4f}, beam rundesh = {args.rbeam or args.beam * 2}")
            save_if_better(ev[0][2], ev[0][0], args.out)
            best = run_rounds(pool, args, P0, rng, ev[0], deadline)
        else:
            if args.start_id is not None:
                starts = [args.start_id]
            elif args.stratified:
                starts = [int(rng.choice(POOLS[r % 3])) for r in range(args.starts)]
            else:
                starts = [int(rng.integers(N)) for _ in range(args.starts)]
            for r, start in enumerate(starts):
                t = time.time()
                print(f"Start {r}: asteroidi {start} (materiali {MATS[start]}), beam={args.beam}")
                ev = evaluate(add_finales(beam_search(pool, [root(start)], args.beam, P0, args.bw), P0))
                if not ev:
                    print("  -> asnje zgjidhje e vlefshme")
                    continue
                saved = save_if_better(ev[0][2], ev[0][0], args.out)
                print(f"  -> raundi 0: objektivi = {ev[0][0]:.4f} | {time.time() - t:.0f}s" + (f" | ruajtur ne {args.out}" if saved else ""))
                res = run_rounds(pool, args, P0, rng, ev[0], deadline)
                if best is None or res[0] < best[0]:
                    best = res

    if best is None:
        print("Asnje zgjidhje e vlefshme.")
        return
    print(f"\nMe e miri ne kete run: {best[0]:.4f}  (lider: -10.687)")
    in_file = file_objective(args.out)
    if in_file is not None:
        print(f"Skedari {args.out} permban tani: {in_file:.4f}")
    U.udp.pretty(best[2])


if __name__ == "__main__":
    main()
