import numpy as np
import matplotlib.pyplot as plt
import util
from util import fpca_plain, fpca_iv, BSpline_plain, BSpline_IV


# =========================================================
# 1. build assigned beta function
# =========================================================
def make_beta_assigned(t):
    """
    你可以自由改這個函數
    """
    return 15 * t * np.exp(-8 * t)


# =========================================================
# 2. generate Y from assigned beta
# =========================================================
def generate_Y_from_beta(W, t, beta_func, noise_std=0.0001):
    """
    W: (n, T)
    t: (T,)
    beta_func: function beta(t)
    """

    beta_true = beta_func(t)

    dt = t[1] - t[0]

    # numerical integral: ∫ W(t) beta(t) dt
    signal = np.trapezoid(W * beta_true[None, :], t, axis=1)

    noise = noise_std * np.random.randn(W.shape[0])

    Y_assigned = signal + noise

    return Y_assigned, beta_true


# =========================================================
# 3. plotting pipeline (same structure as yours)
# =========================================================
def plot_beta_assigned_comparison(
    W, M, Y_assigned, t, beta_assigned, *,
    k_fpca_plain,
    k_fpca_iv,
    k_spline_plain,
    k_spline_iv,
    fig_title="Beta_assigned Recovery Comparison"
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # ===== FPCA Plain =====
    beta_hat, _ = fpca_plain(W, Y_assigned, t, n_components=k_fpca_plain)
    axes[0].plot(t, beta_hat)
    axes[0].plot(t, beta_assigned)
    axes[0].set_title(f"FPCA Plain (k={k_fpca_plain})")
    axes[0].legend(["beta_hat", "beta_assigned"])
    axes[0].grid()

    # ===== FPCA IV =====
    beta_hat, _ = fpca_iv(W, M, Y_assigned, t, n_components=k_fpca_iv)
    axes[1].plot(t, beta_hat)
    axes[1].plot(t, beta_assigned)
    axes[1].set_title(f"FPCA IV (k={k_fpca_iv})")
    axes[1].legend(["beta_hat", "beta_assigned"])
    axes[1].grid()

    # ===== BSpline Plain =====
    beta_hat, _ = BSpline_plain(W, Y_assigned, t, n_components=k_spline_plain)
    axes[2].plot(t, beta_hat)
    axes[2].plot(t, beta_assigned)
    axes[2].set_title(f"BSpline Plain (k={k_spline_plain})")
    axes[2].legend(["beta_hat", "beta_assigned"])
    axes[2].grid()

    # ===== BSpline IV =====
    beta_hat, _ = BSpline_IV(W, M, Y_assigned, t, n_components=k_spline_iv)
    axes[3].plot(t, beta_hat)
    axes[3].plot(t, beta_assigned)
    axes[3].set_title(f"BSpline IV (k={k_spline_iv})")
    axes[3].legend(["beta_hat", "beta_assigned"])
    axes[3].grid()

    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()


# =========================================================
# 4. main experiment
# =========================================================
W, M, Y, t = util.extract_argo_data()

beta_assigned = make_beta_assigned(t)

Y_assigned, _ = generate_Y_from_beta(
    W, t,
    beta_func=make_beta_assigned,
    noise_std=1.0
)

plot_beta_assigned_comparison(
    W, M, Y_assigned, t, beta_assigned,
    k_fpca_plain=6,
    k_fpca_iv=2,
    k_spline_plain=15,
    k_spline_iv=8,
    fig_title=f"Recovery Test (Assigned Beta)"
)