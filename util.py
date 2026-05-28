import numpy as np
import skfda
import xarray as xr
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

def generate_functional_data_GP_SNR(
    n,
    T,
    SNR,
    seed=None
):
    """
    Generate functional regression data using Gaussian Processes
    with Squared Exponential kernels.

    Model
    -----
    X(t) ~ GP(mean_X, K_X)

    W(t) = X(t) + U(t)

    M(t) = X(t) + omega(t)

    Y = integral beta(t) X(t) dt + eps

    SNR := Var(X) / Var(U)
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
    beta_true = 15 * t * np.exp(-8 * t)

    # --------------------------------------------------
    # Mean functions
    # --------------------------------------------------
    mean_X = np.sin(2 * np.pi * t)

    mean_U = np.zeros(T)

    mean_omega = np.zeros(T)

    # --------------------------------------------------
    # Squared Exponential Kernel
    # --------------------------------------------------
    def SE_kernel(t, sigma, length_scale):
        """
        Squared Exponential covariance kernel
        """
        sqdist = (t[:, None] - t[None, :]) ** 2

        return sigma**2 * np.exp(
            -sqdist / (2 * length_scale**2)
        )

    # --------------------------------------------------
    # X process
    # --------------------------------------------------
    sigma_X = 0.1
    ell_X = 0.2

    K_X = SE_kernel(
        t,
        sigma=sigma_X,
        length_scale=ell_X
    )

    X = np.random.multivariate_normal(
        mean=mean_X,
        cov=K_X,
        size=n
    )

    # empirical variance for SNR calibration
    var_X = np.var(X)

    # --------------------------------------------------
    # Measurement error process U
    # --------------------------------------------------
    sigma_U = np.sqrt(var_X / SNR)

    ell_U = 0.1

    K_U = SE_kernel(
        t,
        sigma=sigma_U,
        length_scale=ell_U
    )

    U = np.random.multivariate_normal(
        mean=mean_U,
        cov=K_U,
        size=n
    )

    # --------------------------------------------------
    # IV noise process omega
    # --------------------------------------------------
    sigma_omega = 0.25
    ell_omega = 0.15

    K_omega = SE_kernel(
        t,
        sigma=sigma_omega,
        length_scale=ell_omega
    )

    omega = np.random.multivariate_normal(
        mean=mean_omega,
        cov=K_omega,
        size=n
    )

    # --------------------------------------------------
    # Observed functional processes
    # --------------------------------------------------
    W = X + U

    M = X + omega

    # --------------------------------------------------
    # Scalar response
    # --------------------------------------------------
    sigma_eps = 0.05

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
    print(f"Var(X)      = {var_X:.4f}")
    print(f"sigma_U     = {sigma_U:.4f}")

    return X, W, M, Y, beta_true, t

def generate_functional_data_FPCA_GP(
    n,
    T,
    SNR,
    seed=None
):
    """
    Generate FPCA-friendly functional regression data
    using Gaussian Processes with Squared Exponential kernels.

    Model
    -----
    X(t) ~ GP(mean_X, K_X)

    W(t) = X(t) + U(t)

    M(t) = X(t) + omega(t)

    Y = integral beta(t) X(t) dt + eps

    SNR := Var(X) / Var(U)
    """

    if seed is not None:
        np.random.seed(seed)

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
    t = np.linspace(0, 1, T)

    # --------------------------------------------------
    # True beta(t)
    # --------------------------------------------------
    beta_true = 15 * t * np.exp(-8 * t)

    # --------------------------------------------------
    # Mean function
    # --------------------------------------------------
    mean_X = np.sin(2 * np.pi * t)

    # --------------------------------------------------
    # Squared Exponential Kernel
    # --------------------------------------------------
    def SE_kernel(t, sigma, length_scale):

        sqdist = (
            t[:, None] - t[None, :]
        ) ** 2

        return (
            sigma**2
            * np.exp(
                -sqdist / (2 * length_scale**2)
            )
        )

    # --------------------------------------------------
    # Latent process X
    # --------------------------------------------------
    sigma_X = 0.35
    ell_X = 0.20

    K_X = SE_kernel(
        t,
        sigma=sigma_X,
        length_scale=ell_X
    )

    X = np.random.multivariate_normal(
        mean=mean_X,
        cov=K_X,
        size=n
    )

    # --------------------------------------------------
    # Empirical SNR calibration
    # --------------------------------------------------
    var_X = np.var(X)

    sigma_U = np.sqrt(var_X / SNR)

    # --------------------------------------------------
    # Measurement error process U
    # --------------------------------------------------
    ell_U = 0.08

    K_U = SE_kernel(
        t,
        sigma=sigma_U,
        length_scale=ell_U
    )

    U = np.random.multivariate_normal(
        mean=np.zeros(T),
        cov=K_U,
        size=n
    )

    # --------------------------------------------------
    # Instrument noise process omega
    # --------------------------------------------------
    sigma_omega = 0.25
    ell_omega = 0.12

    K_omega = SE_kernel(
        t,
        sigma=sigma_omega,
        length_scale=ell_omega
    )

    omega = np.random.multivariate_normal(
        mean=np.zeros(T),
        cov=K_omega,
        size=n
    )

    # --------------------------------------------------
    # Observed functional variables
    # --------------------------------------------------
    W = X + U

    M = X + omega

    # --------------------------------------------------
    # Scalar response
    # --------------------------------------------------
    sigma_eps = 0.05

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
    # Theoretical covariance eigensystem
    # (for diagnostics only)
    # --------------------------------------------------
    eigvals, eigvecs = np.linalg.eigh(K_X)

    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    explained = eigvals / eigvals.sum()

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print(f"SNR target = {SNR}")

    print(f"Var(X) = {var_X:.4f}")

    print(f"sigma_U = {sigma_U:.4f}")

    print("\nTop variance ratios:")

    print(explained[:10])

    return (
        X,
        W,
        M,
        Y,
        beta_true,
        t
    )

def generate_functional_data_decreasing_variance_GP(
    n,
    T,
    SNR,
    seed=None,
    length_scale=0.15
):
    """
    Generate functional regression data with:

    - variance decreasing as t increases
    - smooth GP trajectories
    - squared exponential kernel

    Model
    -----
    X_i(t) = mean(t) + sigma(t) * GP_i(t)

    where GP_i(t) ~ GP(0, K)

    Returns
    -------
    X, W, M, Y, beta_true, t
    """

    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
    t = np.linspace(0, 1, T)

    dt = t[1] - t[0]

    # --------------------------------------------------
    # True beta
    # --------------------------------------------------
    beta_true = np.sin(2 * np.pi * t)

    # --------------------------------------------------
    # Mean function
    # --------------------------------------------------
    mean_function = np.sin(2 * np.pi * t)

    # --------------------------------------------------
    # Variance decreases as t increases
    # --------------------------------------------------
    sigma_X_t = 0.5 * (1 - t) + 0.02

    # --------------------------------------------------
    # Squared exponential kernel
    # --------------------------------------------------
    tt1, tt2 = np.meshgrid(t, t)

    K = np.exp(
        - (tt1 - tt2)**2
        / (2 * length_scale**2)
    )

    # --------------------------------------------------
    # Heteroscedastic covariance
    #
    # Cov[X(s), X(t)]
    # = sigma(s) K(s,t) sigma(t)
    # --------------------------------------------------
    Sigma_X = (
        sigma_X_t[:, None]
        * K
        * sigma_X_t[None, :]
    )

    # small jitter for numerical stability
    Sigma_X += 1e-8 * np.eye(T)

    # --------------------------------------------------
    # Generate latent X
    # --------------------------------------------------
    X = np.random.multivariate_normal(
        mean=mean_function,
        cov=Sigma_X,
        size=n
    )

    # empirical Var(X)
    var_X = np.var(X)

    # --------------------------------------------------
    # Back solve sigma_U from SNR
    # --------------------------------------------------
    sigma_U = np.sqrt(var_X / SNR)

    # --------------------------------------------------
    # Measurement error U
    #
    # Also generated from GP
    # --------------------------------------------------
    Sigma_U = (sigma_U**2) * K
    Sigma_U += 1e-8 * np.eye(T)

    U = np.random.multivariate_normal(
        mean=np.zeros(T),
        cov=Sigma_U,
        size=n
    )

    # --------------------------------------------------
    # Instrument noise omega
    # --------------------------------------------------
    sigma_omega = 0.25

    Sigma_omega = (sigma_omega**2) * K
    Sigma_omega += 1e-8 * np.eye(T)

    omega = np.random.multivariate_normal(
        mean=np.zeros(T),
        cov=Sigma_omega,
        size=n
    )

    # --------------------------------------------------
    # Observed processes
    # --------------------------------------------------
    W = X + U
    M = X + omega

    # --------------------------------------------------
    # Scalar response
    # --------------------------------------------------
    signal = np.trapezoid(
        beta_true * X,
        t,
        axis=1
    )

    sigma_eps = 0.05

    Y = signal + np.random.normal(
        0,
        sigma_eps,
        size=n
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print(f"SNR target = {SNR} (Noise @ {100/SNR:.2f}%)")
    print(f"sigma_U set to {sigma_U:.4f}")

    return X, W, M, Y, beta_true, t

def extract_argo_data(
    nc_path=r"E:\argo_program\argo_weekly_profiles_2020_2024.nc",
    weekly_price=None
):
    """
    Extract ARGO functional data and prepare regression variables.

    Parameters
    ----------
    nc_path : str
        Path to NetCDF file.

    weekly_price : array-like or None
        Weekly scalar response variable.
        If provided, will be converted to numpy array and reversed.

    Returns
    -------
    W : ndarray
        Salinity functional data. Shape: (time, depth)

    M : ndarray
        Temperature functional data. Shape: (time, depth)

    Y : ndarray or None
        Scalar response variable.

    t : ndarray
        Normalized depth grid in [0, 1].

    x_space : ndarray
        Same as t.

    weeks : ndarray
        Original time axis from dataset.

    depths : ndarray
        Original depth axis from dataset.

    final_array : ndarray
        Combined array with shape:
        (time, depth, 2)

        final_array[:, :, 0] = TEMP
        final_array[:, :, 1] = PSAL
    """

    # --------------------------------------------------
    # Read dataset
    # --------------------------------------------------
    ds = xr.open_dataset(nc_path)

    # --------------------------------------------------
    # Extract variables
    # --------------------------------------------------
    temp = ds["argo_weekly"].sel(variable="TEMP").values
    psal = ds["argo_weekly"].sel(variable="PSAL").values

    # --------------------------------------------------
    # Coordinates
    # --------------------------------------------------
    weeks = ds["time"].values
    depths = ds["depth"].values

    # --------------------------------------------------
    # Combine into one tensor
    # --------------------------------------------------
    final_array = np.stack((temp, psal), axis=-1)

    print("final_array shape:", final_array.shape)

    # --------------------------------------------------
    # Functional variables
    # --------------------------------------------------
    W = final_array[:, :, 1]   # PSAL
    M = final_array[:, :, 0]   # TEMP

    # --------------------------------------------------
    # Response variable
    # --------------------------------------------------
    if weekly_price is not None:
        Y = np.array(weekly_price[::-1])
    else:
        Y = None

    # --------------------------------------------------
    # Functional grid
    # --------------------------------------------------
    T = W.shape[1]

    x_space = np.linspace(0, 1, T)
    t = np.linspace(0, 1, T)

    return W, M, Y, t, x_space, weeks, depths, final_array

