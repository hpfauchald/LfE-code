import numpy as np
from sim_cohorts import sim_cohorts

def simulate_single_path(
    path_index, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, sigma_S,
    bet, That, Npre, tau, IntVec, Delta_s_t, rlog
):
    """
    Runs a single simulation path of the economy.

    Simulates one realization of the stochastic process governing output, beliefs,
    and asset prices, using the learning-from-experience model. Returns the simulated
    time series and key outputs used for further analysis.

    Parameters
    ----------
    path_index : int
        Index of the simulation path (useful when parallelizing).
    Nt : int
        Number of time steps in the simulation.
    dt : float
        Time increment size (e.g., 1/12 for monthly).
    rho : float
        Time discount rate.
    nu : float
        Death rate or cohort replacement rate.
    Vbar : float
        Prior variance of the belief process.
    mu_Y : float
        Drift of the output process.
    sigma_Y : float
        Diffusion of the output process.
    sigma_S : float
        Diffusion of the stock price process (equals sigma_Y in equilibrium).
    bet : float
        Discounting parameter from pre-calculation.
    That : float
        Length of pre-trading period in years.
    Npre : int
        Number of pre-trading steps.
    tau : np.ndarray
        Initial time-since-entry for each cohort.
    IntVec : np.ndarray
        Initial integration vector for beliefs.
    Delta_s_t : np.ndarray
        Initial belief values of the cohorts.
    rlog : float
        Log risk-free rate (adjusts expected return).

    Returns
    -------
    tuple
        A tuple containing the following:
        - dR : np.ndarray
            Excess returns.
        - Et : np.ndarray
            Market expectation variance over time.
        - Vt : np.ndarray
            Private belief variance over time.
        - Deltabar2 : np.ndarray
            Mean cohort belief over time.
        - r_t : np.ndarray
            Risk-free rate process over time.
        - theta_t : np.ndarray
            Market price of risk over time.
        - Zt : np.ndarray
            Brownian motion path.
        - Port : np.ndarray
            Portfolio allocation (shares in risky asset).
        - mu_S_adj : np.ndarray
            Adjusted expected return under the true measure.
        - mu_S_t_adj : np.ndarray
            Adjusted expected return under each cohort's belief.
        - muhat_S_t_adj : np.ndarray
            Adjusted consensus belief about expected return.
        - muC_s_t : np.ndarray
            Consumption drift.
        - sigmaC_s_t : np.ndarray
            Consumption volatility.
        - f_avg : np.ndarray
            Average belief weight across cohorts.
        - corr_muS_muHat : float
            Correlation between true and consensus beliefs.
    """
    # Generate new shocks
    dZt = np.sqrt(dt) * np.random.randn(Nt)
    Zt = np.cumsum(dZt)
    dZforbias = dZt  # Pre-computed shock series
    biasvec = dZforbias[-Npre:]

    # Simulate this single path
    (
        Xt2, Deltabar2, Part1, mu_S, mu_S_t, muhat_S_t, r_t, theta_t, Port,
        muC_s_t, sigmaC_s_t, BIGf, BIGDELTA, Et, Vt, dR
    ) = sim_cohorts(
        biasvec, dZt, Nt, tau, IntVec, Delta_s_t, dt, rho, nu, Vbar,
        mu_Y, sigma_Y, sigma_S, bet, That, Npre
    )

    # Adjust expected returns
    mu_S_adj = mu_S + rlog - r_t
    mu_S_t_adj = mu_S_t + rlog - r_t
    muhat_S_t_adj = muhat_S_t + rlog - r_t

    # Return results
    return (
        dR, Et, Vt, Deltabar2, r_t, theta_t, Zt[:Nt], Port,
        mu_S_adj, mu_S_t_adj, muhat_S_t_adj, muC_s_t, sigmaC_s_t,
        np.mean(BIGf, axis=0), np.corrcoef(muhat_S_t, mu_S)[0, 1]
    )
