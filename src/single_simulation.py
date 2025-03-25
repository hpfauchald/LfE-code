import numpy as np
from sim_cohorts import sim_cohorts

def simulate_single_path(
    path_index, n_timesteps, dt, rho, nu, Vbar, mu_Y, sigma_Y, sigma_S,
    bet, build_up_period, n_pre_periods, tau, integration_vector, cohort_beliefs, rlog
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
    n_timesteps : int
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
    build_up_period : float
        Length of pre-trading period in years.
    n_pre_periods : int
        Number of pre-trading steps.
    tau : np.ndarray
        Initial time-since-entry for each cohort.
    integration_vector : np.ndarray
        Initial integration vector for beliefs.
    cohort_beliefs : np.ndarray
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
        - avg_belief_series : np.ndarray
            Mean cohort belief over time.
        - r_t : np.ndarray
            Risk-free rate process over time.
        - theta_t : np.ndarray
            Market price of risk over time.
        - z_t : np.ndarray
            Brownian motion path.
        - portfolio_allocation : np.ndarray
            Portfolio allocation (shares in risky asset).
        - mu_s_adj : np.ndarray
            Adjusted expected return under the true measure.
        - mu_s_t_adj : np.ndarray
            Adjusted expected return under each cohort's belief.
        - mu_hat_s_t_adj : np.ndarray
            Adjusted consensus belief about expected return.
        - mu_c_t : np.ndarray
            Consumption drift.
        - sigma_c_t : np.ndarray
            Consumption volatility.
        - weights_avg : np.ndarray
            Average belief weight across cohorts.
        - corr_mu_s_mu_hat : float
            Correlation between true and consensus beliefs.
    """
    # Generate new shocks
    dz_t = np.sqrt(dt) * np.random.randn(n_timesteps)
    z_t = np.cumsum(dz_t)
    dz_for_bias = dz_t  # Pre-computed shock series
    bias_vec = dz_for_bias[-n_pre_periods:]

    # Simulate this single path
    (
        xt_series, avg_belief_series, exp_learning_term, mu_s, mu_s_t, mu_hat_s_t,
        r_t, theta_t, portfolio_allocation, mu_c_t, sigma_c_t,
        weights_matrix, beliefs_matrix, Et, Vt, dR
    ) = sim_cohorts(
        bias_vec, dz_t, n_timesteps, tau, integration_vector, cohort_beliefs, dt,
        rho, nu, Vbar, mu_Y, sigma_Y, sigma_S, bet, build_up_period, n_pre_periods
    )

    # Adjust expected returns
    mu_s_adj = mu_s + rlog - r_t
    mu_s_t_adj = mu_s_t + rlog - r_t
    mu_hat_s_t_adj = mu_hat_s_t + rlog - r_t

    # Return results
    return (
        dR, Et, Vt, avg_belief_series, r_t, theta_t, z_t[:n_timesteps], portfolio_allocation,
        mu_s_adj, mu_s_t_adj, mu_hat_s_t_adj, mu_c_t, sigma_c_t,
        np.mean(weights_matrix, axis=0), np.corrcoef(mu_hat_s_t, mu_s)[0, 1]
    )
