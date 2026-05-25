import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from sklearn.linear_model import LinearRegression

def fpca_plain(W, Y, t, n_components=3, ridge=0):
    """
    FPCA-based scalar-on-function regression (NO IV)

    Parameters
    ----------
    W : (n, T) matrix
    Y : (n,) vector
    t : (T,) grid
    n_components : FPCA components K
    ridge : regularization

    Returns
    -------
    gamma : (K,) coefficient in FPCA basis
    phi_eval : (K, T) eigenfunctions evaluated on grid
    beta_hat : (T,) reconstructed beta(t)
    W_scores : (n, K) FPCA scores
    Y_hat_score : (n,) prediction in score space
    sigma2 : residual variance
    """

    dt = t[1] - t[0]
    n, T = W.shape

    # ========= Step 1: 建立 FDataGrid =========
    W_fd = skfda.FDataGrid(W, grid_points=t)

    # ========= Step 2: FPCA on W =========
    fpca = FPCA(n_components=n_components)
    fpca.fit(W_fd)

    # print("Explained variance ratio sum:",
    #       sum(fpca.explained_variance_ratio_))

    phi = fpca.components_
    phi_eval = phi(t)[:, :, 0]      # (K, T)

    # ========= Step 3: 中心化 =========
    W_center = W - W.mean(axis=0)
    Y_center = Y - Y.mean()

    # ========= Step 4: 計算 FPCA scores =========
    W_scores = W_center @ phi_eval.T * dt   # ξ_ik

    # ========= Step 5: OLS 解 gamma =========
    A = W_scores.T @ W_scores + ridge * np.eye(n_components)
    b = W_scores.T @ Y_center

    gamma = np.linalg.solve(A, b)

    # ========= Step 6: 還原 beta(t) =========
    beta_hat = gamma @ phi_eval

    # ========= Step 7: 預測 =========
    Y_hat_score = W_scores @ gamma

    rss = np.sum((Y_center - Y_hat_score) ** 2)

    sigma2 = rss / n

    K = n_components
    BIC = n * np.log(sigma2) + K * np.log(n)

    R2 = 1 - rss / np.sum((Y_center - Y_center.mean())**2)
    # print("R^2 =", R2)

    return beta_hat, BIC

def fpca_iv(W, M, Y, t, n_components=3, ridge=0, s=1):
    """
    FPCA-based Functional IV Regression
    with normalization (mean 0, std 1)

    Parameters
    ----------
    M : (n, T) matrix
        Instrument functions

    W : (n, T) matrix
        Functional predictors

    Y : (n,) vector
        Scalar response

    t : (T,) grid

    n_components : int
        Number of FPCA components

    ridge : float
        Ridge regularization

    s : float
        Scaling parameter

    Returns
    -------
    beta_hat : (T,)
        Estimated coefficient function

    BIC : float
        Bayesian Information Criterion
    """

    dt = t[1] - t[0]
    n, T = M.shape

    # ==================================================
    # Step 1: Normalize M
    # ==================================================
    M_mean = M.mean(axis=0)
    M_std = 1 # M.std(axis=0)

    # M_std[M_std == 0] = 1

    M_norm = (M - M_mean) / M_std

    # ==================================================
    # Step 2: Normalize W
    # ==================================================
    W_mean = W.mean(axis=0)
    W_std = 1 #W.std(axis=0)

    # W_std[W_std == 0] = 1

    W_norm = (W - W_mean) / W_std

    # ==================================================
    # Step 3: Center Y
    # ==================================================
    Y_center = Y - Y.mean()

    # ==================================================
    # Step 4: Build functional object
    # ==================================================
    W_fd = skfda.FDataGrid(W_norm, grid_points=t)

    # ==================================================
    # Step 5: FPCA on M
    # ==================================================
    fpca = FPCA(n_components=n_components)
    fpca.fit(W_fd)

    phi = fpca.components_
    phi_eval = phi(t)[:, :, 0]    # (K, T)

    # ==================================================
    # Step 6: Compute scores
    # ==================================================
    M_scores = M_norm @ phi_eval.T * dt
    W_scores = W_norm @ phi_eval.T * dt

    # ==================================================
    # Step 7: Covariance in score space
    # ==================================================
    Cov_WM = (W_scores.T @ M_scores) / n
    Cov_YM = (M_scores.T @ Y_center) / n

    # ==================================================
    # Step 8: Solve gamma
    # ==================================================
    A = (s * Cov_WM.T) @ (s * Cov_WM) \
        + ridge * np.eye(n_components)

    b = (s ** 2) * Cov_WM.T @ Cov_YM

    gamma = np.linalg.solve(A, b)

    # ==================================================
    # Step 9: Reconstruct beta(t)
    # ==================================================
    beta_hat = gamma @ phi_eval

    # ==================================================
    # Step 10: Prediction
    # ==================================================
    # Y_hat = np.array([
    #     np.trapezoid(beta_hat * W_norm[i], t)
    #     for i in range(len(W_norm))
    # ])

    # ==================================================
    # Step 11: BIC
    # ==================================================
    Y_hat_score = M_scores @ gamma

    rss = np.sum((Y_center - Y_hat_score) ** 2)

    sigma2 = rss / n

    K = n_components

    BIC = n * np.log(sigma2) + K * np.log(n)

    R2 = 1 - rss / np.sum(
        (Y_center - Y_center.mean()) ** 2
    )

    # print(sum(fpca.explained_variance_ratio_))
    # print("R^2 =", R2)

    return beta_hat, BIC

def BSpline_plain(W, Y, t, n_components=6):

    n, T = W.shape
    dt = t[1] - t[0]

    # B-spline basis
    basis = BSplineBasis(n_basis=n_components)

    # Center Y, W
    Y_center = Y - Y.mean()
    W_center = W - W.mean(axis=0)

    # 取出真正的 (K, T)
    Phi = basis(t)[:, :, 0]

    # =========================
    # Gram correction
    # =========================
    G = Phi @ Phi.T * dt
    L = np.linalg.cholesky(G)
    Phi = np.linalg.solve(L, Phi)

    # 計算 Z = ∫ W(t) φ(t) dt
    Z = W_center @ Phi.T * dt   # (n, K)

    # 做一般線性回歸
    reg = LinearRegression(fit_intercept=False)
    reg.fit(Z, Y_center)

    theta = reg.coef_  # (K,)

    # 重建 beta(t)
    beta_hat = theta @ Phi  # (T,)

    # Prediction
    Y_hat = reg.predict(Z)
    rss = np.sum((Y_center - Y_hat) ** 2)

    k = n_components
    bic = n * np.log(rss / n) + k * np.log(n)

    return beta_hat, bic

def BSpline_IV(W, M, Y, t, n_components=4, ridge=0):

    n, T = W.shape

    dt = t[1] - t[0]

    # ==================================================
    # Step 1: Normalize W
    # ==================================================
    W_mean = W.mean(axis=0)
    W_std = 1

    W_norm = (W - W_mean) / W_std

    # ==================================================
    # Step 2: Normalize M
    # ==================================================
    M_mean = M.mean(axis=0)
    M_std = 1

    M_norm = (M - M_mean) / M_std

    # ==================================================
    # Step 3: Center Y
    # ==================================================
    Y_center = Y - Y.mean()

    # ==================================================
    # Step 4: Build B-spline basis
    # ==================================================
    basis = BSplineBasis(n_basis=n_components)

    phi_eval = basis.evaluate(t)[:, :, 0]

    # =========================
    # Gram correction
    # =========================
    G = phi_eval @ phi_eval.T * dt
    L = np.linalg.cholesky(G)
    phi_eval = np.linalg.solve(L, phi_eval)

    # ==================================================
    # Step 5: Projection scores
    # ==================================================
    M_scores = M_norm @ phi_eval.T * dt
    W_scores = W_norm @ phi_eval.T * dt

    # ==================================================
    # Step 6: Covariance in score space
    # ==================================================
    Cov_WM = (W_scores.T @ M_scores) / n
    Cov_YM = (M_scores.T @ Y_center) / n

    # ==================================================
    # Step 7: Solve gamma
    # ==================================================
    A = Cov_WM.T @ Cov_WM + ridge * np.eye(n_components)
    b = Cov_WM.T @ Cov_YM

    gamma = np.linalg.solve(A, b)

    # ==================================================
    # Step 8: Reconstruct beta(t)
    # ==================================================
    beta_hat = gamma @ phi_eval

    # ==================================================
    # Step 9: Prediction
    # ==================================================
    Y_hat = W_scores @ gamma

    rss = np.sum((Y_center - Y_hat) ** 2)

    # ==================================================
    # Step 10: BIC
    # ==================================================
    bic = n * np.log(rss / n) + n_components * np.log(n)

    return beta_hat, bic

def generate_functional_data_SNR(n, T, SNR, seed=None):
    """
    Generate functional regression data where SNR controls measurement error U,
    exactly matching the IV paper setting.

    SNR := Var(X) / Var(U)

    Fixed:
        sigma_eps   = 0.05   (regression noise)
        sigma_omega = 0.25   (IV noise)
    """

    if seed is not None:
        np.random.seed(seed)

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
    t = np.linspace(0, 1, T)

    # --------------------------------------------------
    # True beta
    # --------------------------------------------------
    beta_true = np.sin(2 * np.pi * t)

    # --------------------------------------------------
    # Latent X(t)
    # --------------------------------------------------
    sigma_X = 0.1
    X = np.sin(2*np.pi*t)[None, :] + np.random.normal(0, sigma_X, size=(n, T))

    # empirical Var(X) for correct SNR control
    var_X = np.var(X)

    # --------------------------------------------------
    # Back-solve sigma_U from SNR
    # --------------------------------------------------
    sigma_U = np.sqrt(var_X / SNR)

    # --------------------------------------------------
    # Noises
    # --------------------------------------------------
    sigma_eps = 0.05
    sigma_omega = 0.25

    U = np.random.normal(0, sigma_U, size=(n, T))
    omega = np.random.normal(0, sigma_omega, size=(n, T))

    # --------------------------------------------------
    # Observed processes
    # --------------------------------------------------
    W = X + U
    M = X + omega   # IV: correlated with X but independent of U

    # --------------------------------------------------
    # Response
    # --------------------------------------------------
    signal = np.trapezoid(beta_true * X, t, axis=1)
    Y = signal + np.random.normal(0, sigma_eps, size=n)

    print(f"SNR target = {SNR}(Noise @ {100*1/SNR}%)")
    print(f"sigma_U set to {sigma_U:.4f}")
    # print(f"sigma_eps fixed at {sigma_eps}")
    # print(f"sigma_omega fixed at {sigma_omega}")

    return X, W, M, Y, beta_true, t

def generate_functional_data_FPCA(
    n,
    T,
    SNR,
    seed=None,
    K_true=3
):
    """
    Generate FPCA-friendly functional regression data.

    Model:
        X_i(t) = sum_k a_ik phi_k(t)

    SNR := Var(X) / Var(U)

    Returns
    -------
    X, W, M, Y, beta_true, t
    """

    if seed is not None:
        np.random.seed(seed)

    if K_true > 7:
        raise ValueError(
            "Current implementation only supports K_true <= 7"
        )

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
    t = np.linspace(0, 1, T)
    dt = t[1] - t[0]

    # --------------------------------------------------
    # True beta(t)
    # --------------------------------------------------
    beta_true = (
        1.5 * np.sin(2 * np.pi * t)
        + 0.5 * np.cos(4 * np.pi * t)
    )

    # --------------------------------------------------
    # True eigenfunctions phi_k(t)
    # orthonormal-ish basis
    # --------------------------------------------------
    phi1 = np.sqrt(2) * np.sin(np.pi * t)
    phi2 = np.sqrt(2) * np.cos(2 * np.pi * t)
    phi3 = np.sqrt(2) * np.sin(3 * np.pi * t)
    phi4 = np.sqrt(2) * np.cos(4 * np.pi * t)
    phi5 = np.sqrt(2) * np.sin(5 * np.pi * t)
    phi6 = np.sqrt(2) * np.cos(6 * np.pi * t)
    phi7 = np.sqrt(2) * np.sin(7 * np.pi * t)

    Phi = np.vstack([
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi6,
        phi7
    ])[:K_true]

    # --------------------------------------------------
    # FPCA scores
    # decreasing eigenvalues
    # --------------------------------------------------
    lambdas = np.array([
        1.5,
        0.7,
        0.3,
        0.15,
        0.08,
        0.04,
        0.02
    ])[:K_true]

    scores = np.random.normal(
        0,
        np.sqrt(lambdas),
        size=(n, K_true)
    )

    # --------------------------------------------------
    # Construct latent X(t)
    # --------------------------------------------------
    X = scores @ Phi

    # small smooth residual variation
    X += 0.05 * np.random.normal(size=(n, T))

    # --------------------------------------------------
    # SNR calibration
    # --------------------------------------------------
    var_X = np.var(X)

    sigma_U = np.sqrt(var_X / SNR)

    # --------------------------------------------------
    # Noises
    # --------------------------------------------------
    sigma_eps = 0.05
    sigma_omega = 0.25

    U = np.random.normal(0, sigma_U, size=(n, T))
    omega = np.random.normal(0, sigma_omega, size=(n, T))

    # --------------------------------------------------
    # Observed functional variables
    # --------------------------------------------------
    W = X + U

    # IV process
    M = X + omega

    # --------------------------------------------------
    # Scalar response
    # --------------------------------------------------
    signal = np.trapezoid(
        beta_true * X,
        t,
        axis=1
    )

    Y = signal + np.random.normal(
        0,
        sigma_eps,
        size=n
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print(f"SNR target = {SNR}")
    print(f"sigma_U = {sigma_U:.4f}")
    print(f"Var(X) = {var_X:.4f}")

    explained = lambdas / np.sum(lambdas)

    print("True variance ratios:")
    print(explained)

    return X, W, M, Y, beta_true, t

