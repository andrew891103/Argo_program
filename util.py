import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from sklearn.linear_model import LinearRegression

def fpca_plain(W, Y, t, n_components=3, ridge=0.0):
    """
    Correct FPCA-based scalar-on-function regression
    using proper functional inner products.
    """

    n, T = W.shape
    dt = t[1] - t[0]

    # ==================================================
    # Step 1: Center W and Y
    # ==================================================
    W_center = W - W.mean(axis=0)
    Y_center = Y - Y.mean()

    # ==================================================
    # Step 2: FPCA to get eigenfunctions
    # ==================================================
    W_fd = skfda.FDataGrid(W_center, grid_points=t)

    fpca = FPCA(n_components=n_components)
    fpca.fit(W_fd)

    # eigenfunctions φ_k(t)  → (K, T)
    phi = fpca.components_(t)[:, :, 0]

    # ==================================================
    # Step 3: Compute TRUE functional scores
    #   ξ_ik = ∫ W_i(t) φ_k(t) dt
    # ==================================================
    W_scores = W_center @ phi.T * dt   # (n, K)

    # ==================================================
    # Step 4: OLS in score space
    # ==================================================
    A = W_scores.T @ W_scores + ridge * np.eye(n_components)
    b = W_scores.T @ Y_center

    gamma = np.linalg.solve(A, b)

    # ==================================================
    # Step 5: Reconstruct beta(t)
    #   β(t) = Σ γ_k φ_k(t)
    # ==================================================
    beta_hat = gamma @ phi  # (T,)

    # ==================================================
    # Step 6: Prediction using integral form
    # ==================================================
    Y_hat = W_scores @ gamma
    rss = np.sum((Y_center - Y_hat) ** 2)

    sigma2 = rss / n
    BIC = n * np.log(sigma2) + n_components * np.log(n)

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

    #Center Y, W
    Y_center = Y - Y.mean()
    W_center = W - W.mean(axis=0)

    # 取出真正的 (K, T)
    Phi = basis(t)[:, :, 0]

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
    """
    B-spline Functional IV Regression
    with normalization (mean 0, std 1)

    Parameters
    ----------
    X : (n, T) matrix
        Functional predictors

    M : (n, T) matrix
        Instrument functions

    Y : (n,) vector
        Scalar response

    t : (T,) grid

    n_components : int
        Number of B-spline basis functions

    ridge : float
        Ridge regularization

    Returns
    -------
    beta_hat : (T,)
        Estimated coefficient function

    bic : float
        Bayesian Information Criterion
    """

    n, T = W.shape

    dt = t[1] - t[0]

    # ==================================================
    # Step 1: Normalize X
    # ==================================================
    W_mean = W.mean(axis=0)
    W_std = 1 #X.std(axis=0)

    # X_std[X_std == 0] = 1

    W_norm = (W - W_mean) / W_std

    # ==================================================
    # Step 2: Normalize M
    # ==================================================
    M_mean = M.mean(axis=0)
    M_std = 1 # M.std(axis=0)

    # M_std[M_std == 0] = 1

    M_norm = (M - M_mean) / M_std

    # ==================================================
    # Step 3: Center Y
    # ==================================================
    Y_center = Y - Y.mean()

    # ==================================================
    # Step 4: Build B-spline basis
    # ==================================================
    basis = BSplineBasis(n_basis=n_components)

    # (K, T)
    phi_eval = basis.evaluate(t)[:, :, 0]

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
    A = Cov_WM.T @ Cov_WM \
        + ridge * np.eye(n_components)

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
    bic = n * np.log(rss / n) \
          + n_components * np.log(n)

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



