import util
from util import generate_functional_data_FPCA, generate_functional_data_SNR

import matplotlib.pyplot as plt
import numpy as np

def plot_functional_data(
    X,
    W,
    M,
    beta_true,
    t,
    n_curves=20
):
    """
    Visualize functional regression data.
    """

    n = X.shape[0]

    idx = np.random.choice(n, n_curves, replace=False)

    # ==========================================
    # Latent X
    # ==========================================
    plt.figure(figsize=(8, 4))

    for i in idx:
        plt.plot(t, X[i], alpha=0.7)

    plt.title("Latent Functional Process X(t)")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.grid(True)
    plt.show()

    # ==========================================
    # Observed W
    # ==========================================
    plt.figure(figsize=(8, 4))

    for i in idx:
        plt.plot(t, W[i], alpha=0.7)

    plt.title("Observed Process W(t) = X(t) + U(t)")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.show()

    # ==========================================
    # Instrument M
    # ==========================================
    plt.figure(figsize=(8, 4))

    for i in idx:
        plt.plot(t, M[i], alpha=0.7)

    plt.title("Instrument Process M(t)")
    plt.xlabel("t")
    plt.ylabel("M(t)")
    plt.grid(True)
    plt.show()

    # ==========================================
    # True beta
    # ==========================================
    plt.figure(figsize=(8, 4))

    plt.plot(t, beta_true, linewidth=3)

    plt.title("True Beta Function")
    plt.xlabel("t")
    plt.ylabel(r"$\beta(t)$")
    plt.grid(True)
    plt.show()

X, W, M, Y, beta_true, t = generate_functional_data_SNR(
    n=500,
    T=100,
    SNR=10,
    seed=0
)

plot_functional_data(
    X, W, M,
    beta_true,
    t,
    n_curves=20
)
