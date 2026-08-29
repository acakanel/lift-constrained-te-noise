#!/usr/bin/env python3
"""Brooks-Pope-Marcolini (NASA RP-1218) trailing-edge and separated-flow self-noise, implemented complete."""
import numpy as np

C0 = 340.46
THIRD_OCT = np.array([200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
                      2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
                      20000], float)


def _Amin(a):
    a = abs(a)
    if a <= 0.204: return np.sqrt(max(67.552 - 886.788 * a * a, 0)) - 8.219
    if a < 0.244:  return -32.665 * a + 3.981
    return -142.795 * a ** 3 + 103.656 * a * a - 57.757 * a + 6.006


def _Amax(a):
    a = abs(a)
    if a <= 0.13: return np.sqrt(max(67.552 - 886.788 * a * a, 0)) - 8.219
    if a < 0.321: return -15.901 * a + 1.098
    return -4.669 * a ** 3 + 3.491 * a * a - 16.699 * a + 1.149


def _a0(Re):
    if Re < 9.52e4:  return 0.57
    if Re <= 8.57e5: return -9.57e-13 * (Re - 8.57e5) ** 2 + 1.13
    return 1.13


def _A(St_ratio, Re):
    a = abs(np.log10(max(St_ratio, 1e-12)))
    a0 = _a0(Re)
    AR = (-20 - _Amin(a0)) / (_Amax(a0) - _Amin(a0))
    return _Amin(a) + AR * (_Amax(a) - _Amin(a))


def _Bmin(b):
    b = abs(b)
    if b <= 0.13: return np.sqrt(max(16.888 - 886.788 * b * b, 0)) - 4.109
    if b < 0.145: return -83.607 * b + 8.138
    return -817.810 * b ** 3 + 355.210 * b * b - 135.024 * b + 10.619


def _Bmax(b):
    b = abs(b)
    if b <= 0.10: return np.sqrt(max(16.888 - 886.788 * b * b, 0)) - 4.109
    if b < 0.187: return -31.330 * b + 1.854
    return -80.541 * b ** 3 + 44.174 * b * b - 39.381 * b + 2.344


def _b0(Re):
    if Re < 9.52e4:  return 0.30
    if Re <= 8.57e5: return -4.48e-13 * (Re - 8.57e5) ** 2 + 0.56
    return 0.56


def _B(St_ratio, Re):
    b = abs(np.log10(max(St_ratio, 1e-12)))
    b0 = _b0(Re)
    BR = (-20 - _Bmin(b0)) / (_Bmax(b0) - _Bmin(b0))
    return _Bmin(b) + BR * (_Bmax(b) - _Bmin(b))


def _K1(Re):
    if Re < 2.47e5:  return -4.31 * np.log10(Re) + 156.3
    if Re <= 8.0e5:  return -9.0 * np.log10(Re) + 181.6
    return 128.5


def _gammas(M):
    return (27.094 * M + 3.31,
            23.43 * M + 4.651,
            72.65 * M + 10.74,
            -34.19 * M - 13.82)


def _K2(alpha, Re, M):
    g, g0, be, be0 = _gammas(M)
    K1 = _K1(Re)
    if alpha < (g0 - g):
        return K1 - 1000.0
    if alpha <= (g0 + g):
        return K1 + np.sqrt(max(be * be - (be / g) ** 2 * (alpha - g0) ** 2, 0)) + be0
    return K1 - 12.0


def _dK1(alpha, Re_dp):
    if Re_dp <= 5000:
        return alpha * (1.43 * np.log10(max(Re_dp, 1.0)) - 5.29)
    return 0.0


def alpha_switch(M):
    return min(_gammas(M)[1], 12.5)


def _Dh(theta, phi, M):
    Mc = 0.8 * M
    num = 2 * np.sin(theta / 2) ** 2 * np.sin(phi) ** 2
    den = (1 + M * np.cos(theta)) * (1 + (M - Mc) * np.cos(theta)) ** 2
    return num / max(den, 1e-12)


def _Dl(theta, phi, M):
    return np.sin(theta) ** 2 * np.sin(phi) ** 2 / max((1 + M * np.cos(theta)) ** 4, 1e-12)


def spl_third_octave(dstar_s, dstar_p, U, alpha_deg, chord, span=1.0, nu=1.5e-5,
                     r=1.0, theta=np.pi / 2, phi=np.pi / 2, freqs=THIRD_OCT,
                     return_components=False):
    freqs = np.atleast_1d(np.asarray(freqs, float))
    M = U / C0
    Rec = U * chord / nu
    Redp = U * dstar_p / nu
    Dh, Dl = _Dh(theta, phi, M), _Dl(theta, phi, M)

    St1 = 0.02 * M ** (-0.6)
    a = abs(alpha_deg)
    if a < 1.33:     f2 = 1.0
    elif a <= 12.5:  f2 = 10 ** (0.0054 * (a - 1.33) ** 2)
    else:            f2 = 4.72
    St2 = St1 * f2
    St1_bar = 0.5 * (St1 + St2)

    K1 = _K1(Rec); K2 = _K2(a, Rec, M); dK1 = _dK1(a, Redp)
    stalled = a >= alpha_switch(M)

    base_h_s = 10 * np.log10(dstar_s * M ** 5 * span * Dh / max(r * r, 1e-12))
    base_h_p = 10 * np.log10(dstar_p * M ** 5 * span * Dh / max(r * r, 1e-12))
    base_l_s = 10 * np.log10(dstar_s * M ** 5 * span * Dl / max(r * r, 1e-12))

    tot = np.zeros(len(freqs)); comp = {"p": [], "s": [], "alpha": []}
    for i, f in enumerate(freqs):
        Sts = f * dstar_s / U; Stp = f * dstar_p / U
        if stalled:
            SPL_p = SPL_s = -np.inf
            SPL_a = base_l_s + _A(Sts / St2, 3.0 * Rec) + K2
        else:
            SPL_p = base_h_p + _A(Stp / St1, Rec) + (K1 - 3) + dK1
            SPL_s = base_h_s + _A(Sts / St1_bar, Rec) + (K1 - 3)
            SPL_a = base_h_s + _B(Sts / St2, Rec) + K2
        tot[i] = 10 * np.log10(sum(10 ** (x / 10) for x in (SPL_p, SPL_s, SPL_a)
                                   if np.isfinite(x)) + 1e-300)
        comp["p"].append(SPL_p); comp["s"].append(SPL_s); comp["alpha"].append(SPL_a)

    if return_components:
        return tot, {k: np.array(v) for k, v in comp.items()}, dict(
            stalled=stalled, St1=St1, St2=St2, St1_bar=St1_bar, K1=K1, K2=K2,
            dK1=dK1, alpha_switch=alpha_switch(M), Dh=Dh, Dl=Dl)
    return tot


def overall_spl(dstar_s, dstar_p, U, alpha_deg, chord, **kw):
    s = spl_third_octave(dstar_s, dstar_p, U, alpha_deg, chord, **kw)
    return 10 * np.log10(np.sum(10 ** (s / 10)))


def oaspl_a_weighted(dstar_s, dstar_p, U, alpha_deg, chord, freqs=THIRD_OCT, **kw):
    f = np.asarray(freqs, float)
    ra = (12194.0 ** 2 * f ** 4) / ((f ** 2 + 20.6 ** 2) *
          np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) * (f ** 2 + 12194.0 ** 2))
    A = 20 * np.log10(ra) + 2.00
    s = spl_third_octave(dstar_s, dstar_p, U, alpha_deg, chord, freqs=f, **kw)
    return 10 * np.log10(np.sum(10 ** ((s + A) / 10)))
