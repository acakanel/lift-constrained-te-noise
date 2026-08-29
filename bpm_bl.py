#!/usr/bin/env python3
"""Brooks–Pope–Marcolini (NASA RP-1218) boundary-layer thickness correlations for a NACA 0012 section."""
import numpy as np

def dstar0_over_c(Rec, tripped=True):
    Rec = np.asarray(Rec, float)
    lg = np.log10(Rec)
    if tripped:
        low = 0.0601 * Rec ** (-0.114)
        high = 10.0 ** (3.411 - 1.5397 * lg + 0.1059 * lg ** 2)
        return np.where(Rec <= 0.3e6, low, high)
    return 10.0 ** (3.0187 - 1.5397 * lg + 0.1059 * lg ** 2)


def delta0_over_c(Rec, tripped=True):
    lg = np.log10(np.asarray(Rec, float))
    a = 1.892 if tripped else 1.6569
    return 10.0 ** (a - 0.9045 * lg + 0.0596 * lg ** 2)


def theta0_over_c(Rec, tripped=True):
    Rec = np.asarray(Rec, float)
    lg = np.log10(Rec)
    if tripped:
        low = 0.0723 * Rec ** (-0.1765)
        high = 10.0 ** (0.5578 - 0.7079 * lg + 0.0404 * lg ** 2)
        return np.where(Rec <= 0.3e6, low, high)
    return 10.0 ** (0.2021 - 0.7079 * lg + 0.0404 * lg ** 2)


def ratio_pressure(alpha):
    a = np.abs(np.asarray(alpha, float))
    return 10.0 ** (-0.0432 * a + 0.00113 * a ** 2)


def ratio_suction(alpha, tripped=True):
    a = np.abs(np.asarray(alpha, float))
    if tripped:
        return np.where(a < 5.0, 10.0 ** (0.0679 * a),
               np.where(a <= 12.5, 0.381 * 10.0 ** (0.1516 * a),
                                   14.296 * 10.0 ** (0.0258 * a)))
    return np.where(a < 7.5, 10.0 ** (0.0679 * a),
           np.where(a <= 12.5, 0.0162 * 10.0 ** (0.3066 * a),
                               54.42 * 10.0 ** (0.0258 * a)))


def dstar_both_sides(alpha, Rec, chord, tripped=True):
    d0 = dstar0_over_c(Rec, tripped) * chord
    return d0 * ratio_suction(alpha, tripped), d0 * ratio_pressure(alpha)


def dstar_p_from_dstar_s(dstar_s, alpha, tripped=True):
    d0 = np.asarray(dstar_s, float) / ratio_suction(alpha, tripped)
    return d0 * ratio_pressure(alpha)


def infer_tripped(dstar_s_meas, alpha, Rec, chord):
    err = {}
    for trip in (True, False):
        pred = dstar0_over_c(Rec, trip) * chord * ratio_suction(alpha, trip)
        err[trip] = abs(np.log(pred / dstar_s_meas))
    return err[True] <= err[False]
