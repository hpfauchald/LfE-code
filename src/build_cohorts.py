import numpy as np
from postvar import post_var  # Import the posterior variance function

# This function initializes a large number of cohorts before simulation.
def build_up_cohorts(
    dz_t: np.ndarray,
    n_timesteps: int,
    dt: float,
    rho: float,
    nu: float,
    prior_variance: float,
    mu_Y: float,
    sigma_Y: float,
    bet: float,
    build_up_period: float
):
    """
    Simulates the build-up phase of a large number of cohorts before running the main economic model.

    This function generates the initial distribution of beliefs, outputs, and state variables 
    based on a learning-from-experience framework. It models how agents update their beliefs 
    about the economy using Bayesian updating, given prior variance and observed shocks.

    Parameters:
    -----------
    dz_t : np.ndarray
        Array of Brownian motion increments (shocks), shape (n_timesteps - 1,).
    n_timesteps : int
        Total number of time steps for simulation.
    dt : float
        Time step size (e.g., 1/12 for monthly data).
    rho : float
        Time discount rate.
    nu : float
        Death rate or rate of cohort replacement.
    prior_variance : float
        Prior variance of beliefs about the state variable.
    mu_Y : float
        Growth rate of output.
    sigma_Y : float
        Standard deviation of output shocks.
    bet : float
        Equilibrium parameter computed from model pre-calculations.
    build_up_period : float
        Length of the pre-trading (build-up) period.

    Returns:
    --------
    avg_belief : np.ndarray
        Average belief (mean of belief distribution) at each time step, shape (n_timesteps,).
    integration_vector : np.ndarray
        Integration vector capturing weighted beliefs used for updating, shape varies over time.
    Xt : np.ndarray
        Aggregated weight on beliefs at each time step, shape (n_timesteps,).
    cohort_beliefs : np.ndarray
        Belief about the state variable (e.g., drift term) at each time step, shape varies over time.
    Yt : np.ndarray
        Simulated output (e.g., GDP or income), shape (n_timesteps,).
    Zt : np.ndarray
        Simulated Brownian motion path (state shocks), shape (n_timesteps,).
    normalized_weights : np.ndarray
        Normalized integration vector (belief distribution weights), shape varies over time.
    tau : np.ndarray
        Time vector for each belief cohort (time since each cohort entered the economy), shape varies over time.
    """

    n_pre_periods = int(build_up_period / dt)  # Pre-trading period (converted to an integer for indexing)

    Zt = np.insert(np.cumsum(dz_t), 0, 0)  # Brownian motion simulation
    yg = (mu_Y - 0.5 * sigma_Y**2) * dt + sigma_Y * dz_t  # Log-growth rate process
    Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)  # Simulated output

    Xt = np.ones(n_timesteps)
    avg_belief = np.zeros(n_timesteps)
    cohort_beliefs = np.array([0.0])  # Initialized as a numpy array

    integration_vector = np.array([nu * bet])  # Integration vector
    tau = np.array([dt])  # Time vector

    reduction = np.exp(-nu * dt)  # Discounting factor

    for i in range(1, n_timesteps):
        Part = integration_vector * np.exp(
            -(rho + 0.5 * cohort_beliefs**2) * dt + cohort_beliefs * dz_t[i - 1]
        )

        Xt[i] = np.sum(Part)
        avg_belief[i] = np.sum(Part * cohort_beliefs) / Xt[i]

        integration_vector = np.append(reduction * Part, bet * (1 - reduction) * Xt[i])  # Update integration vector
        normalized_weights = integration_vector / Xt[i]  # Normalize

        # Compute belief updates using the posterior variance function
        d_cohort_belief = (
            post_var(sigma_Y, prior_variance, tau) / sigma_Y**2
        ) * (-cohort_beliefs * dt + dz_t[i - 1])

        if i < (n_pre_periods + 1):
            cohort_beliefs = np.append(cohort_beliefs + d_cohort_belief, 0)
        else:
            bias_adjustment = np.sum(dz_t[i - n_pre_periods : i]) / build_up_period
            cohort_beliefs = np.append(cohort_beliefs + d_cohort_belief, bias_adjustment)

        tau = np.append(tau + dt, 0)  # Update time vector

    return avg_belief, integration_vector, Xt, cohort_beliefs, Yt, Zt, normalized_weights, tau
