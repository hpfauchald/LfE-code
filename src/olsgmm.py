import numpy as np
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw
import scipy.stats as stats

def olsgmm(lhv: np.ndarray, rhv: np.ndarray, lags: int, weight: int):
    """
    Performs OLS regression with GMM-corrected standard errors.

    Parameters:
    lhv (np.ndarray): Left-hand variable (dependent variable), shape (T, N).
    rhv (np.ndarray): Right-hand variables (independent variables), shape (T, K).
    lags (int): Number of lags for GMM standard error correction.
    weight (int): Weighting scheme for standard errors (-1: Skip, 0: Equal, 1: Newey-West).

    Returns:
    bv (np.ndarray): OLS regression coefficients.
    sebv (np.ndarray): Standard errors of coefficients.
    R2v (float): Unadjusted R².
    R2vadj (float): Adjusted R².
    v (np.ndarray): Variance-covariance matrix.
    F (np.ndarray): Chi-square test results for joint significance.
    """

    # Ensure inputs are NumPy arrays
    lhv = np.asarray(lhv)
    rhv = np.asarray(rhv)

    T, N = lhv.shape  # Number of observations, number of dependent variables
    K = rhv.shape[1]  # Number of independent variables

    # OLS regression: bv = (X'X)^(-1) X'Y
    bv = np.linalg.lstsq(rhv, lhv, rcond=None)[0]

    if weight == -1:
        return bv, np.nan, np.nan, np.nan, np.nan, np.nan

    # Compute residuals
    errv = lhv - rhv @ bv

    # Compute variance of residuals
    s2 = np.mean(errv**2, axis=0)
    vary = np.mean((lhv - np.mean(lhv, axis=0))**2, axis=0)

    # Compute R² and adjusted R²
    R2v = (1 - s2 / vary)
    R2vadj = (1 - (s2 / vary) * (T - 1) / (T - K))

    # Compute robust standard errors using Newey-West estimator
    sebv = np.zeros((K, N))
    v_list = []
    F = np.zeros((N, 3))  # Stores Chi-square test results

    for i in range(N):
        model = sm.OLS(lhv[:, i], rhv).fit()
        
        if weight == 1:
            # Newey-West HAC covariance estimator
            varb = sw.cov_hac(model, nlags=lags)
        else:
            # Homoskedastic variance-covariance
            varb = model.cov_params()

        # Compute standard errors
        seb = np.sqrt(np.diag(varb))
        sebv[:, i] = seb

        # Store variance-covariance matrix
        v_list.append(varb)

        # Chi-square test for joint significance (excluding constant)
        if np.all(rhv[:, 0] == 1):  # If first column is a constant
            chi2_val = bv[1:, i].T @ np.linalg.inv(varb[1:, 1:]) @ bv[1:, i]
            dof = K - 1  # Exclude constant
        else:
            chi2_val = bv[:, i].T @ np.linalg.inv(varb) @ bv[:, i]
            dof = K

        pval = 1 - stats.chi2.cdf(chi2_val, dof)
        F[i, :] = [chi2_val, dof, pval]

    v = np.vstack(v_list)  # Stack variance matrices

    return bv, sebv, R2v, R2vadj, v, F
