import matplotlib.pyplot as plt
from util import fpca_plain, fpca_iv, BSpline_plain, BSpline_IV
import util

def plot_beta_comparison(
    W, M, Y, t,
    *,
    k_fpca_plain,
    k_fpca_iv,
    k_spline_plain,
    k_spline_iv,
    fig_title="Beta(t) Estimation Comparison"
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # ===== FPCA Plain =====
    beta_hat, _ = fpca_plain(W, Y, t, n_components=k_fpca_plain)
    ax = axes[0]
    ax.plot(t, beta_hat)
    ax.set_title(f"FPCA Plain (k={k_fpca_plain})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.grid()

    # ===== FPCA IV =====
    beta_hat, _ = fpca_iv(W, M, Y, t, n_components=k_fpca_iv)
    ax = axes[1]
    ax.plot(t, beta_hat)
    ax.set_title(f"FPCA + IV (k={k_fpca_iv})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.grid()

    # ===== BSpline Plain =====
    beta_hat, _ = BSpline_plain(W, Y, t, n_components=k_spline_plain)
    ax = axes[2]
    ax.plot(t, beta_hat)
    ax.set_title(f"BSpline Plain (k={k_spline_plain})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.grid()

    # ===== BSpline IV =====
    beta_hat, _ = BSpline_IV(W, M, Y, t, n_components=k_spline_iv)
    ax = axes[3]
    ax.plot(t, beta_hat)
    ax.set_title(f"BSpline + IV (k={k_spline_iv})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.grid()

    fig.suptitle(fig_title, fontsize=16)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    plt.show()

W, M, Y, t = util.extract_argo_data()

plot_beta_comparison(
    W, M, Y, t,
    k_fpca_plain=6,
    k_fpca_iv=2,
    k_spline_plain=15,
    k_spline_iv=8,
    fig_title=f"Beta Recovery of Argo Data"
)

