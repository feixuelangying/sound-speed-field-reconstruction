# -*- coding: utf-8 -*-
"""物理换算：声速方程（Chen-Millero/UNESCO 主）、Mackenzie 对照、TEOS-10 gsw 对照；
压力-深度换算（gsw，z 取负向约定）。全部接受 numpy 数组。"""
import numpy as np
import seawater as sw
import gsw


def sound_speed(T, S, p, eq="chen-millero"):
    """T: degC(ITS-90), S: psu(PSS-78), p: dbar -> 声速 m/s"""
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    p = np.asarray(p, dtype=float)
    if eq == "chen-millero":
        return sw.svel(S, T, p)
    if eq == "gsw":
        return gsw.sound_speed(S, T, p)
    if eq == "mackenzie":
        # Mackenzie (1981) 九项方程，自变量为深度 D(m，正值)
        D = z_from_p(p, lat=20.0)
        c = (1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
             + 1.340 * (S - 35.0) + 1.630e-2 * D + 1.675e-7 * D**2
             - 1.025e-2 * T * (S - 35.0) - 7.139e-13 * T * D**3)
        return c
    raise ValueError(eq)


def z_from_p(p, lat):
    """压力(dbar) -> 深度(m，正值向下)。gsw 约定 z 为负，此处返回正深度。"""
    return -gsw.z_from_p(np.asarray(p, dtype=float), np.asarray(lat, dtype=float))


def p_from_z(z, lat):
    """深度(m，正值向下) -> 压力(dbar)。"""
    return gsw.p_from_z(-np.asarray(z, dtype=float), np.asarray(lat, dtype=float))


def sound_speed_jacobian(T, S, p, eq="chen-millero", hT=0.05, hS=0.05, hP=5.0):
    """中心差分声速 Jacobian（∂c/∂T, ∂c/∂S, ∂c/∂p），供 u_eq 的 LPU 传播。"""
    T = np.asarray(T, dtype=float); S = np.asarray(S, dtype=float); p = np.asarray(p, dtype=float)
    dc_dT = (sound_speed(T + hT, S, p, eq) - sound_speed(T - hT, S, p, eq)) / (2 * hT)
    dc_dS = (sound_speed(T, S + hS, p, eq) - sound_speed(T, S - hS, p, eq)) / (2 * hS)
    dc_dp = (sound_speed(T, S, p + hP, eq) - sound_speed(T, S, p - hP, eq)) / (2 * hP)
    return dc_dT, dc_dS, dc_dp
