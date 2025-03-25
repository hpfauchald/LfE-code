import numpy as np
from postvar import post_var  # Import posterior variance function

def sim_cohorts(
    bias_vec: np.ndarray,
    dz_t: np.ndarray,
    n_timesteps: int,
    tau: np.ndarray,
    integration_vector: np.ndarray,
    cohort_beliefs: np.ndarray,
    dt: float,
    rho: float,
    nu: float,
    Vbar: float,
    mu_Y: float,
    sigma_Y: float,
    sigma_S: float,
    bet: float,
    build_up_period: float,
    n_pre_periods: int
):
    """
    Simulates the forward dynamics of the economy with overlapping cohorts under 
    a learning-from-experience framework.

    This function propagates agents’ beliefs and economic variables forward in time 
    given initial conditions and stochastic shocks, tracking the evolution of beliefs, 
    expected returns, consumption dynamics, and other macro-financial variables across cohorts.

    Parameters
    ----------
    bias_vec : np.ndarray
        Vector of initial bias adjustments based on past shocks, length n_pre_periods.
    dz_t : np.ndarray
        Brownian motion increments (shocks), shape (n_timesteps,).
    n_timesteps : int
        Total number of simulation steps.
    tau : np.ndarray
        Vector of cohort ages (time since entry), used to compute posterior variance.
    integration_vector : np.ndarray
        Initial weight vector used in belief aggregation.
    cohort_beliefs : np.ndarray
        Initial vector of cohort-specific beliefs (e.g., about a drift term).
    dt : float
        Time step size (e.g., 1/12 for monthly frequency).
    rho : float
        Time preference rate.
    nu : float
        Death or cohort replacement rate.
    Vbar : float
        Prior belief variance.
    mu_Y : float
        Drift of the output process.
    sigma_Y : float
        Volatility of the output process.
    sigma_S : float
        Volatility of the asset return process.
    bet : float
        Steady-state consumption share parameter.
    build_up_period : float
        Length of the pre-trading (bias estimation) period in years.
    n_pre_periods : int
        Number of periods used for estimating the initial bias.

    Returns
    -------
    cohort_weight_sum : np.ndarray
        Sum of belief weights across cohorts at each time, shape (n_timesteps,).
    avg_cohort_belief : np.ndarray
        Weighted average belief across cohorts at each time, shape (n_timesteps,).
    experience_term : np.ndarray
        Experience-based moving average of shocks, shape (n_timesteps,).
    mu_S : np.ndarray
        Expected excess return under the true measure, shape (n_timesteps,).
    mu_S_t : np.ndarray
        Expected excess return under the subjective measure, shape (n_timesteps,).
    muhat_S_t : np.ndarray
        Cross-sectional average of cohort beliefs about expected return, shape (n_timesteps,).
    r_t : np.ndarray
        Risk-free rate at each time, shape (n_timesteps,).
    theta_t : np.ndarray
        Sharpe ratio (price of risk) at each time, shape (n_timesteps,).
    portfolio_choice : np.ndarray
        Optimal portfolio weights at each time step, shape (n_timesteps,).
    muC_s_t : np.ndarray
        Drift of log consumption growth across cohorts, shape (n_timesteps,).
    sigmaC_s_t : np.ndarray
        Volatility of log consumption growth across cohorts, shape (n_timesteps,).
    weight_matrix : np.ndarray
        Normalized cohort weight matrix over time, shape (n_timesteps, n_timesteps).
    belief_matrix : np.ndarray
        Cohort belief matrix over time, shape (n_timesteps, n_timesteps).
    belief_expectation : np.ndarray
        Cohort-weighted expectation of belief deviation, shape (n_timesteps,).
    belief_variance : np.ndarray
        Cross-sectional variance of belief deviations, shape (n_timesteps,).
    dR : np.ndarray
        Total return (including dividends) on the risky asset, shape (n_timesteps,).
    """

    # Precompute posterior variance based on tau (cohort specific)
    post_var_vec = post_var(sigma_Y, Vbar, tau)

    # --- Preallocating arrays ---
    cohort_weight_sum = np.zeros(n_timesteps)
    avg_cohort_belief = np.zeros(n_timesteps)
    belief_expectation = np.zeros(n_timesteps)
    belief_variance = np.zeros(n_timesteps)
    experience_term = np.zeros(n_timesteps)
    dR = np.zeros(n_timesteps)
    belief_matrix = np.zeros((n_timesteps, n_timesteps))
    weight_matrix = np.zeros((n_timesteps, n_timesteps))

    # Reverse weights
    reduction = np.exp(-nu * dt)
    RevNt = np.arange(n_timesteps, 0, -1)
    death_weight_vector = (reduction ** (RevNt - 1)) * nu * dt
    death_weight_vector /= np.sum(death_weight_vector)  # Normalize

    # Other preallocated arrays
    mu_S = np.zeros(n_timesteps)
    mu_S_t = np.zeros(n_timesteps)
    muhat_S_t = np.zeros(n_timesteps)
    muC_s_t = np.zeros(n_timesteps)
    sigmaC_s_t = np.zeros(n_timesteps)
    r_t = np.zeros(n_timesteps)
    theta_t = np.zeros(n_timesteps)
    current_cohort_weight = np.zeros(n_timesteps)

    bias_weight = (Vbar / (1 + (Vbar / sigma_Y**2) * dt * RevNt)) * (1 / sigma_Y)

    # Loop over time steps
    for i in range(n_timesteps):
        # Update integral vector and normalized weights
        Part = integration_vector * np.exp(-0.5 * cohort_beliefs**2 * dt + cohort_beliefs * dz_t[i])
        cohort_weight_sum[i] = np.sum(Part)
        avg_cohort_belief[i] = np.sum(Part * cohort_beliefs) / cohort_weight_sum[i]
        f = Part / cohort_weight_sum[i]

        # Belief updating for experience term (part1)
        if i == 0:
            experience_term[i] = np.sum(bias_vec) / build_up_period
        else:
            experience_term[i] = experience_term[i - 1] + (
                post_var(sigma_Y, Vbar, i * dt) / sigma_Y**2
            ) * (-experience_term[i - 1] * dt + dz_t[i - 1])

        # Dividend return
        dR[i] = (
            mu_S[i - 1]
            - r_t[i - 1]
            + rho
            + mu_Y
            - sigma_Y**2
            + nu * (1 - bet)
        ) * dt + sigma_S * dz_t[i]

        # Store current Delta and f
        belief_matrix[i, :] = cohort_beliefs
        weight_matrix[i, :] = f

        # Compute expected returns
        mu_S[i] = sigma_S * sigma_Y - (sigma_S - sigma_Y) * avg_cohort_belief[i]
        mu_S_t[i] = sigma_S * sigma_Y + sigma_S * (
            -avg_cohort_belief[i] + experience_term[i]
        ) + sigma_Y * avg_cohort_belief[i]
        muhat_S_t[i] = mu_S[i] + sigma_S * death_weight_vector @ cohort_beliefs

        # Risk-free rate and price of risk
        r_t[i] = rho + mu_Y - sigma_Y**2 + nu * (1 - bet) + sigma_Y * avg_cohort_belief[i]
        theta_t[i] = sigma_Y - avg_cohort_belief[i]

        # Expectation and variance of beliefs
        belief_expectation[i] = f @ bias_weight
        belief_variance[i] = (f @ cohort_beliefs**2) * sigma_Y - avg_cohort_belief[i]**2 * sigma_Y

        current_cohort_weight[i] = f[n_timesteps - (i + 1)]

        # Consumption drift and volatility
        muC_s_t[i] = mu_Y + nu * (1 - bet) + (
            sigma_Y - avg_cohort_belief[i]
        ) * (experience_term[i] - avg_cohort_belief[i])
        sigmaC_s_t[i] = sigma_Y + experience_term[i] - avg_cohort_belief[i]

        # --- Belief deviations update (optimized using precomputed post_var_vec) ---
        dDelta_s_t = (post_var_vec / sigma_Y**2) * (-cohort_beliefs * dt + dz_t[i])

        # Bias adjustment depending on i
        if i < n_pre_periods:
            bias_adjustment = (np.sum(bias_vec[i:]) + np.sum(dz_t[:i])) / build_up_period
        else:
            bias_adjustment = np.sum(dz_t[i - n_pre_periods + 1 : i + 1]) / build_up_period

        cohort_beliefs = np.append(cohort_beliefs[1:] + dDelta_s_t[1:], bias_adjustment)
        integration_vector = np.append(reduction * Part[1:], bet * (1 - reduction) * cohort_weight_sum[i])

    # Portfolio choice (final computation)
    portfolio_choice = (
        (experience_term - avg_cohort_belief) / sigma_S
        + (sigma_Y / sigma_S) * (1 - bet * np.flip(death_weight_vector) / current_cohort_weight)
    )

    return (
        cohort_weight_sum,
        avg_cohort_belief,
        experience_term,
        mu_S,
        mu_S_t,
        muhat_S_t,
        r_t,
        theta_t,
        portfolio_choice,
        muC_s_t,
        sigmaC_s_t,
        weight_matrix,
        belief_matrix,
        belief_expectation,
        belief_variance,
        dR,
    )
