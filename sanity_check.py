import numpy as np
import matplotlib.pyplot as plt
import util

from util import (
    fpca_plain,
    fpca_iv,
    BSpline_plain,
    BSpline_IV
)


# =========================================================
# 1. assigned beta
# =========================================================
def make_beta_assigned(t):

    return 15 * t * np.exp(-8 * t)


# =========================================================
# 2. generate Y from assigned beta
# =========================================================
def generate_Y_from_beta(
    W,
    t,
    beta_func,
    noise_std=0.0001
):

    beta_true = beta_func(t)

    signal = np.trapezoid(
        W * beta_true[None, :],
        t,
        axis=1
    )

    noise = noise_std * np.random.randn(W.shape[0])

    Y_assigned = signal + noise

    return Y_assigned, beta_true


# =========================================================
# 3. automatic BIC selection
# =========================================================
def select_best_k_by_BIC(
    W,
    M,
    Y,
    t,
    *,
    fpca_plain_max_k=20,
    fpca_iv_max_k=20,
    spline_plain_max_k=20,
    spline_iv_max_k=20,
    show_plot=True
):

    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    # =====================================================
    # FPCA Plain
    # =====================================================
    k_range = range(1, fpca_plain_max_k + 1)

    bics = []

    for k in k_range:

        _, bic = fpca_plain(
            W,
            Y,
            t,
            n_components=k
        )

        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(k_range)[best_idx]

    results["fpca_plain"] = best_k

    ax = axes[0]

    ax.plot(k_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx])

    ax.set_title(f"FPCA Plain (best k={best_k})")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.grid()

    # =====================================================
    # FPCA IV
    # =====================================================
    k_range = range(1, fpca_iv_max_k + 1)

    bics = []

    for k in k_range:

        _, bic = fpca_iv(
            W,
            M,
            Y,
            t,
            n_components=k
        )

        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(k_range)[best_idx]

    results["fpca_iv"] = best_k

    ax = axes[1]

    ax.plot(k_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx])

    ax.set_title(f"FPCA IV (best k={best_k})")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.grid()

    # =====================================================
    # BSpline Plain
    # =====================================================
    k_range = range(4, spline_plain_max_k + 1)

    bics = []

    for k in k_range:

        _, bic = BSpline_plain(
            W,
            Y,
            t,
            n_components=k
        )

        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(k_range)[best_idx]

    results["spline_plain"] = best_k

    ax = axes[2]

    ax.plot(k_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx])

    ax.set_title(f"BSpline Plain (best k={best_k})")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.grid()

    # =====================================================
    # BSpline IV
    # =====================================================
    k_range = range(4, spline_iv_max_k + 1)

    bics = []

    for k in k_range:

        _, bic = BSpline_IV(
            W,
            M,
            Y,
            t,
            n_components=k
        )

        bics.append(bic)

    best_idx = np.argmin(bics)
    best_k = list(k_range)[best_idx]

    results["spline_iv"] = best_k

    ax = axes[3]

    ax.plot(k_range, bics, marker='o')
    ax.axvline(best_k, linestyle='--')
    ax.scatter(best_k, bics[best_idx])

    ax.set_title(f"BSpline IV (best k={best_k})")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.grid()

    fig.suptitle("BIC Selection")

    fig.subplots_adjust(
        hspace=0.35,
        wspace=0.25
    )

    if show_plot:
        plt.show()

    return results


# =========================================================
# 4. recovery comparison
# =========================================================
def plot_beta_assigned_comparison(
    W,
    M,
    Y_assigned,
    t,
    beta_assigned,
    *,
    k_fpca_plain,
    k_fpca_iv,
    k_spline_plain,
    k_spline_iv,
    fig_title="Recovery Test"
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # =====================================================
    # FPCA Plain
    # =====================================================
    beta_hat, _ = fpca_plain(
        W,
        Y_assigned,
        t,
        n_components=k_fpca_plain
    )

    axes[0].plot(t, beta_hat)
    axes[0].plot(t, beta_assigned)

    axes[0].set_title(
        f"FPCA Plain (k={k_fpca_plain})"
    )

    axes[0].legend([
        "beta_hat",
        "beta_assigned"
    ])

    axes[0].grid()

    # =====================================================
    # FPCA IV
    # =====================================================
    beta_hat, _ = fpca_iv(
        W,
        M,
        Y_assigned,
        t,
        n_components=k_fpca_iv
    )

    axes[1].plot(t, beta_hat)
    axes[1].plot(t, beta_assigned)

    axes[1].set_title(
        f"FPCA IV (k={k_fpca_iv})"
    )

    axes[1].legend([
        "beta_hat",
        "beta_assigned"
    ])

    axes[1].grid()

    # =====================================================
    # BSpline Plain
    # =====================================================
    beta_hat, _ = BSpline_plain(
        W,
        Y_assigned,
        t,
        n_components=k_spline_plain
    )

    axes[2].plot(t, beta_hat)
    axes[2].plot(t, beta_assigned)

    axes[2].set_title(
        f"BSpline Plain (k={k_spline_plain})"
    )

    axes[2].legend([
        "beta_hat",
        "beta_assigned"
    ])

    axes[2].grid()

    # =====================================================
    # BSpline IV
    # =====================================================
    beta_hat, _ = BSpline_IV(
        W,
        M,
        Y_assigned,
        t,
        n_components=k_spline_iv
    )

    axes[3].plot(t, beta_hat)
    axes[3].plot(t, beta_assigned)

    axes[3].set_title(
        f"BSpline IV (k={k_spline_iv})"
    )

    axes[3].legend([
        "beta_hat",
        "beta_assigned"
    ])

    axes[3].grid()

    fig.suptitle(fig_title)

    fig.subplots_adjust(
        hspace=0.35,
        wspace=0.25
    )

    plt.show()


# =========================================================
# 5. main pipeline
# =========================================================
W, M, Y, t = util.extract_argo_data()

# ---------------------------------------------------------
# build assigned beta
# ---------------------------------------------------------
beta_assigned = make_beta_assigned(t)

# ---------------------------------------------------------
# generate artificial Y
# ---------------------------------------------------------
Y_assigned, _ = generate_Y_from_beta(
    W,
    t,
    beta_func=make_beta_assigned,
    noise_std=0.01
)

# ---------------------------------------------------------
# automatic BIC selection
# ---------------------------------------------------------
best_k_dict = select_best_k_by_BIC(
    W,
    M,
    Y_assigned,
    t,
    fpca_plain_max_k=20,
    fpca_iv_max_k=20,
    spline_plain_max_k=20,
    spline_iv_max_k=20
)

print(best_k_dict)

# ---------------------------------------------------------
# recovery comparison
# ---------------------------------------------------------
plot_beta_assigned_comparison(
    W,
    M,
    Y_assigned,
    t,
    beta_assigned,

    k_fpca_plain=best_k_dict["fpca_plain"],
    k_fpca_iv=best_k_dict["fpca_iv"],
    k_spline_plain=best_k_dict["spline_plain"],
    k_spline_iv=best_k_dict["spline_iv"],

    fig_title="Assigned Beta Recovery"
)