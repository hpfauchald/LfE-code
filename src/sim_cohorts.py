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
    # Posterior variance of beliefs based on cohort age τ, using Equation (2.7) from the paper
    post_var_vec = post_var(sigma_Y, Vbar, tau)

    # Preallocate time series arrays for simulation outputs.
    # These will track the evolution of cohort beliefs, weights, returns, and learning over time.
    # Variables like avg_cohort_belief and cohort_weight_sum correspond to the cross-sectional 
    # aggregation of beliefs from Equation (2.6): Δ̄ₜ = ∫ Δ_τ,ₜ μ(dτ)
    cohort_weight_sum = np.zeros(n_timesteps)      # ∫ μ(dτ), denominator of (2.6)
    avg_cohort_belief = np.zeros(n_timesteps)      # Δ̄ₜ in (2.6), cross-sectional average belief
    belief_expectation = np.zeros(n_timesteps)     # ∫ Δ_τ,ₜ μ(dτ), used for tracking forecast deviations
    belief_variance = np.zeros(n_timesteps)        # Cross-sectional variance of beliefs
    experience_term = np.zeros(n_timesteps)        # ℰₜ from Equation (2.8), experience-based learning
    dR = np.zeros(n_timesteps)                     # Realized return on the risky asset
    belief_matrix = np.zeros((n_timesteps, n_timesteps))  # History of Δ_τ,ₜ over time
    weight_matrix = np.zeros((n_timesteps, n_timesteps))  # History of μ(dτ) over time


    # Reverse weights
    # Construct cohort weight vector μ(dτ) for Equation (2.6).
    # μ(dτ) = ν e^(−ντ) dτ is approximated discretely by `death_weight_vector`, which 
    # reflects the exponential distribution of cohort survival over time.
    reduction = np.exp(-nu * dt)
    RevNt = np.arange(n_timesteps, 0, -1)
    death_weight_vector = (reduction ** (RevNt - 1)) * nu * dt
    death_weight_vector /= np.sum(death_weight_vector)  # Normalize

    # Other preallocated arrays
    # Initialize storage for endogenous variables computed in each period.
    # - mu_S, mu_S_t, muhat_S_t relate to expected returns (see Eq. 2.4, 2.5, and 2.6).
    # - r_t is the risk-free rate (Eq. 2.8), theta_t is the Sharpe ratio (Eq. 2.9).
    # - muC_s_t and sigmaC_s_t are drift and volatility of log consumption growth (Eqs. 2.10 and 2.11).
    # - current_cohort_weight tracks weight of the most recent cohort (used in final portfolio expression).
    # - bias_weight is the cross-sectional expectation term in belief updating (used in Equation 2.7).
    mu_S = np.zeros(n_timesteps)
    mu_S_t = np.zeros(n_timesteps)
    muhat_S_t = np.zeros(n_timesteps)
    muC_s_t = np.zeros(n_timesteps)
    sigmaC_s_t = np.zeros(n_timesteps)
    r_t = np.zeros(n_timesteps)
    theta_t = np.zeros(n_timesteps)
    current_cohort_weight = np.zeros(n_timesteps)

    bias_weight = (Vbar / (1 + (Vbar / sigma_Y**2) * dt * RevNt)) * (1 / sigma_Y)

    # Belief aggregation across cohorts (Equations 2.2 and 2.3):
    # - Part is the unnormalized cohort-specific weight based on Bayes' rule (Eq. 2.2).
    # - cohort_weight_sum[i] corresponds to the denominator of Eq. (2.3), normalizing the weights.
    # - avg_cohort_belief[i] implements Eq. (2.3), computing the weighted average belief across cohorts.
    # - f is the normalized belief weight vector for the current period.
    for i in range(n_timesteps):
        # Update integral vector and normalized weights
        Part = integration_vector * np.exp(-0.5 * cohort_beliefs**2 * dt + cohort_beliefs * dz_t[i])
        cohort_weight_sum[i] = np.sum(Part)
        avg_cohort_belief[i] = np.sum(Part * cohort_beliefs) / cohort_weight_sum[i]
        f = Part / cohort_weight_sum[i]

        # Experience-based belief update (Equation 2.5):
        # - For t = 0, initialize the experience term with the average of past shocks.
        # - For t ≥ 1, update using the discretized version of Eq. (2.5),
        #   a mean-reverting process where the posterior variance scales the innovation.
        # - This term reflects how agents form beliefs based on their own lifetime experience of shocks.
        if i == 0:
            experience_term[i] = np.sum(bias_vec) / build_up_period
        else:
            experience_term[i] = experience_term[i - 1] + (
                post_var(sigma_Y, Vbar, i * dt) / sigma_Y**2
            ) * (-experience_term[i - 1] * dt + dz_t[i - 1])

        # Dividend-inclusive return on the risky asset (discretized version of Equation 2.10):
        # - The return reflects expected excess return from the previous period,
        #   adjusted for interest rate, consumption growth, and demographic effects.
        # - The diffusion term sigma_S * dz_t[i] represents the instantaneous shock.
        # - Euler discretization of continuous-time return dynamics with overlapping generations.
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

        # Compute expected returns:
        # Equation (2.5) – Expected excess return under the true measure
        mu_S[i] = sigma_S * sigma_Y - (sigma_S - sigma_Y) * avg_cohort_belief[i]

        # Equation (2.9) – Subjective expected excess return under cohort beliefs
        mu_S_t[i] = sigma_S * sigma_Y + sigma_S * (
            -avg_cohort_belief[i] + experience_term[i]
        ) + sigma_Y * avg_cohort_belief[i]

        # Equation (2.11) – Cross-sectional average of subjective beliefs (μ̂^S_t)
        muhat_S_t[i] = mu_S[i] + sigma_S * death_weight_vector @ cohort_beliefs

        # Risk-free rate and price of risk
        # Equation (2.4) – Risk-free rate
        r_t[i] = rho + mu_Y - sigma_Y**2 + nu * (1 - bet) + sigma_Y * avg_cohort_belief[i]

        # Equation (2.6) – Sharpe ratio (θ_t)
        theta_t[i] = sigma_Y - avg_cohort_belief[i]

        # Expectation and variance of beliefs
        belief_expectation[i] = f @ bias_weight
        belief_variance[i] = (f @ cohort_beliefs**2) * sigma_Y - avg_cohort_belief[i]**2 * sigma_Y

        # Used in the portfolio choice expression (Equation (2.17))
        current_cohort_weight[i] = f[n_timesteps - (i + 1)]

        # Compute the drift and volatility of log consumption growth
        # These expressions are derived from Equation (2.13)
        muC_s_t[i] = mu_Y + nu * (1 - bet) + (
            sigma_Y - avg_cohort_belief[i]
        ) * (experience_term[i] - avg_cohort_belief[i])
        sigmaC_s_t[i] = sigma_Y + experience_term[i] - avg_cohort_belief[i]

        # --- Belief deviations update (optimized using precomputed post_var_vec) ---
        dDelta_s_t = (post_var_vec / sigma_Y**2) * (-cohort_beliefs * dt + dz_t[i])

        # Update the belief of the newly entering cohort based on experience (Equation 2.8)
        if i < n_pre_periods:
            bias_adjustment = (np.sum(bias_vec[i:]) + np.sum(dz_t[:i])) / build_up_period
        else:
            bias_adjustment = np.sum(dz_t[i - n_pre_periods + 1 : i + 1]) / build_up_period

        cohort_beliefs = np.append(cohort_beliefs[1:] + dDelta_s_t[1:], bias_adjustment)
        integration_vector = np.append(reduction * Part[1:], bet * (1 - reduction) * cohort_weight_sum[i])

    # Compute portfolio choice (Equation 2.9)
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
