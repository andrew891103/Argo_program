import numpy as np
import matplotlib.pyplot as plt
from util import fpca_plain, fpca_iv, BSpline_plain, BSpline_IV
import util

def plot_bic_selection(
    W, M, Y, t, *,
    fpca_plain_max_k=20,
    fpca_iv_max_k=20,
    spline_plain_max_k=20,
    spline_iv_max_k=20,
    fig_title="BIC Model Selection"
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    # ====== FPCA Plain ======
    fpca_plain_range = range(1, fpca_plain_max_k + 1)
    bics = []
    for k in fpca_plain_range:
        _, bic = fpca_plain(W, Y, t, n_components=k)
        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(fpca_plain_range)[best_idx]

    ax = axes[0]
    ax.plot(fpca_plain_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx], zorder=5)
    ax.set_title(f"FPCA Plain (best k={best_k})")
    ax.set_xlabel("n_components")
    ax.set_ylabel("BIC")
    ax.grid()

    # ====== FPCA IV ======
    fpca_iv_range = range(1, fpca_iv_max_k + 1)
    bics = []
    for k in fpca_iv_range:
        _, bic = fpca_iv(W, M, Y, t, n_components=k)
        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(fpca_iv_range)[best_idx]

    ax = axes[1]
    ax.plot(fpca_iv_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx], zorder=5)
    ax.set_title(f"FPCA IV (best k={best_k})")
    ax.set_xlabel("n_components")
    ax.set_ylabel("BIC")
    ax.grid()

    # ====== BSpline Plain ======
    spline_plain_range = range(4, spline_plain_max_k + 1)
    bics = []
    for k in spline_plain_range:
        _, bic = BSpline_plain(W, Y, t, n_components=k)
        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(spline_plain_range)[best_idx]

    ax = axes[2]
    ax.plot(spline_plain_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx], zorder=5)
    ax.set_title(f"BSpline Plain (best k={best_k})")
    ax.set_xlabel("n_components")
    ax.set_ylabel("BIC")
    ax.grid()

    # ====== BSpline IV ======
    spline_iv_range = range(4, spline_iv_max_k + 1)
    bics = []
    for k in spline_iv_range:
        _, bic = BSpline_IV(W, M, Y, t, n_components=k)
        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(spline_iv_range)[best_idx]

    ax = axes[3]
    ax.plot(spline_iv_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx], zorder=5)
    ax.set_title(f"BSpline IV (best k={best_k})")
    ax.set_xlabel("n_components")
    ax.set_ylabel("BIC")
    ax.grid()

    # ⭐ 關鍵：解決上下擠在一起
    fig.suptitle(fig_title, fontsize=16)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    plt.show()

SNR = 1

X, W, M, Y, beta_true, t = util.generate_functional_data_SNR(
    n=30000, T=100, SNR=SNR
)

plot_bic_selection(
    W, M, Y, t,
    fpca_plain_max_k=20,
    fpca_iv_max_k=20,
    spline_plain_max_k=20,
    spline_iv_max_k=20,
    fig_title=f"BIC Selection under SNR={SNR}"
)