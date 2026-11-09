"""Floating-point tolerance for squared L2 scores (paper Sec. III-C, EI).

eps = gamma_{d+2}(u_engine) * B_engine + gamma_{d+2}(2^-53) * B_model
  B_engine: magnitude term of the engine's formula
    direct:    sum_i (q_i - v_i)^2
    expansion: ||q||^2 + ||v||^2 + 2 sum_i |q_i v_i|
  B_model:   the model's own float64 direct sum.
Engines that report the distance itself (sqrt) are compared on the square of the
reported value, with the rounding of the square root added (_sqrt_slack).
gamma_n(u) = n u / (1 - n u) bounds the rounding error of an inner product of length n
(Higham 2002; Ipsen and Zhou 2019). The extra 2 covers the subtraction and the square.
"""
import numpy as np

U64 = 2.0 ** -53


def gamma(n, u):
    nu = n * u
    if nu >= 1:
        raise ValueError("n*u must be below 1")
    return nu / (1.0 - nu)


def distances_and_eps(q, V, contract):
    """Exact-model distances from q to each row of V, and the tolerance for each.

    q: (d,) float32, V: (m, d) float32. Returns (delta, eps), both float64 (m,).
    """
    q64 = np.asarray(q, dtype=np.float64)
    V64 = np.asarray(V, dtype=np.float64)
    d = q64.shape[0]
    diff = V64 - q64[None, :]
    delta = np.einsum("ij,ij->i", diff, diff)
    sc = contract["score"]
    g_e = gamma(d + 2, sc["unit_roundoff"])
    g_m = gamma(d + 2, U64)
    if sc["formula"] == "direct":
        b_e = delta
    elif sc["formula"] == "expansion":
        b_e = q64 @ q64 + np.einsum("ij,ij->i", V64, V64) + 2.0 * (np.abs(V64) @ np.abs(q64))
    else:
        raise ValueError(sc["formula"])
    return delta, _sqrt_slack(sc, g_e * b_e + g_m * delta, delta)


def batch_distances_and_eps(Qrows, Vrows, contract):
    """Row-wise version: distance and tolerance between Qrows[i] and Vrows[i]."""
    Q64 = np.asarray(Qrows, dtype=np.float64)
    V64 = np.asarray(Vrows, dtype=np.float64)
    d = Q64.shape[1]
    diff = V64 - Q64
    delta = np.einsum("ij,ij->i", diff, diff)
    sc = contract["score"]
    g_e = gamma(d + 2, sc["unit_roundoff"])
    g_m = gamma(d + 2, U64)
    if sc["formula"] == "direct":
        b_e = delta
    else:
        b_e = (np.einsum("ij,ij->i", Q64, Q64) + np.einsum("ij,ij->i", V64, V64)
               + 2.0 * np.einsum("ij,ij->i", np.abs(Q64), np.abs(V64)))
    return delta, _sqrt_slack(sc, g_e * b_e + g_m * delta, delta)


def _sqrt_slack(sc, eps, delta):
    """Engines that report the L2 distance (a square root) instead of its square: the
    adapter squares the reported value in float64. The square root, rounded with unit
    roundoff u_s, and its square add a relative error below 3 u_s; 4 u_s is used, on
    the largest value the engine's sum can take (delta + eps)."""
    if sc.get("reported") != "sqrt":
        return eps
    return eps + 4.0 * sc["sqrt_unit_roundoff"] * (delta + eps)
