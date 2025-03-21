import numpy as np

# Computes the posterior variance in a Kalman filtering setup when learning about a constant.
def post_var(sig_y: float, v_bar: float, tau: np.ndarray) -> np.ndarray:

    """
    Computes the posterior variance in a Kalman filtering setup when learning about a constant state variable.

    This function returns the posterior variance of an agent's belief about a constant unknown parameter, 
    given prior variance, signal noise variance, and the time since the agent started observing signals. 
    It follows from Bayesian updating under Gaussian assumptions and is used to update beliefs over time.

    Parameters:
    -----------
    sig_y : float
        Standard deviation of the observed signal (noise in the learning process).
    v_bar : float
        Prior variance of the unknown constant (initial uncertainty).
    tau : np.ndarray
        Array of time durations since each cohort started observing the signal (learning periods).

    Returns:
    --------
    np.ndarray
        Posterior variance of the belief about the constant parameter, updated for each time period.
    """

    return (sig_y**2 * v_bar) / (sig_y**2 + v_bar * tau)
