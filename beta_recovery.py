import matplotlib.pyplot as plt
from util import fpca_plain, fpca_iv, BSpline_plain, BSpline_IV
import util

def plot_beta_comparison(
    W, M, Y, t, beta_true, *,
    k_fpca_plain,
    k_fpca_iv,
    k_spline_plain,
    k_spline_iv,
    fig_title="Beta(t) Recovery Comparison"
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # ===== FPCA Plain =====
    beta_hat, _ = fpca_plain(W, Y, t, n_components=k_fpca_plain)
    ax = axes[0]
    ax.plot(t, beta_hat)
    ax.plot(t, beta_true)
    ax.set_title(f"FPCA Plain (k={k_fpca_plain})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.legend(['beta_hat', 'beta_true'])
    ax.grid()

    # ===== FPCA IV =====
    beta_hat, _ = fpca_iv(M, W, Y, t, n_components=k_fpca_iv)
    ax = axes[1]
    ax.plot(t, beta_hat)
    ax.plot(t, beta_true)
    ax.set_title(f"FPCA + IV (k={k_fpca_iv})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.legend(['beta_hat', 'beta_true'])
    ax.grid()

    # ===== BSpline Plain =====
    beta_hat, _ = BSpline_plain(W, Y, t, n_components=k_spline_plain)
    ax = axes[2]
    ax.plot(t, beta_hat)
    ax.plot(t, beta_true)
    ax.set_title(f"BSpline Plain (k={k_spline_plain})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.legend(['beta_hat', 'beta_true'])
    ax.grid()

    # ===== BSpline IV =====
    beta_hat, _ = BSpline_IV(M, W, Y, t, n_components=k_spline_iv)
    ax = axes[3]
    ax.plot(t, beta_hat)
    ax.plot(t, beta_true)
    ax.set_title(f"BSpline + IV (k={k_spline_iv})")
    ax.set_xlabel("t")
    ax.set_ylabel("beta(t)")
    ax.legend(['beta_hat', 'beta_true'])
    ax.grid()

    fig.suptitle(fig_title, fontsize=16)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    plt.show()

SNR = 50

X, W, M, Y, beta_true, t = util.generate_functional_data_SNR(
    n=20000, T=100, SNR=SNR
)

plot_beta_comparison(
    W, M, Y, t, beta_true,
    k_fpca_plain=1,
    k_fpca_iv=1,
    k_spline_plain=4,
    k_spline_iv=4,
    fig_title=f"Beta Recovery under SNR={SNR}"
)

# beta_hat1, _ = fpca_plain(W, Y, t, n_components=99)
# beta_hat2, _ = fpca_iv(M, W, Y, t, n_components=99)
# print("beta_hat1,",beta_hat1,'/n',"beta_hat2,",beta_hat2)

