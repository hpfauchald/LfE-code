import numpy as np
from postvar import post_var  # Import the posterior variance function

# This function initializes a large number of cohorts before simulation.
def build_up_cohorts(dZt: np.ndarray, Nt: int, dt: float, rho: float, nu: float, 
                      Vbar: float, mu_Y: float, sigma_Y: float, bet: float, That: float):
    
    """
    Simulates the build-up phase of a large number of cohorts before running the main economic model.

    This function generates the initial distribution of beliefs, outputs, and state variables 
    based on a learning-from-experience framework. It models how agents update their beliefs 
    about the economy using Bayesian updating, given prior variance and observed shocks.

    Parameters:
    -----------
    dZt : np.ndarray
        Array of Brownian motion increments (shocks), shape (Nt-1,).
    Nt : int
        Total number of time steps for simulation.
    dt : float
        Time step size (e.g., 1/12 for monthly data).
    rho : float
        Time discount rate.
    nu : float
        Death rate or rate of cohort replacement.
    Vbar : float
        Prior variance of beliefs about the state variable.
    mu_Y : float
        Growth rate of output.
    sigma_Y : float
        Standard deviation of output shocks.
    bet : float
        Equilibrium parameter computed from model pre-calculations.
    That : float
        Length of the pre-trading (build-up) period.

    Returns:
    --------
    Deltabar : np.ndarray
        Average belief (mean of belief distribution) at each time step, shape (Nt,).
    IntVec : np.ndarray
        Integration vector capturing weighted beliefs used for updating, shape varies over time.
    Xt : np.ndarray
        Aggregated weight on beliefs at each time step, shape (Nt,).
    Delta_s_t : np.ndarray
        Belief about the state variable (e.g., drift term) at each time step, shape varies over time.
    Yt : np.ndarray
        Simulated output (e.g., GDP or income), shape (Nt,).
    Zt : np.ndarray
        Simulated Brownian motion path (state shocks), shape (Nt,).
    f : np.ndarray
        Normalized integration vector (belief distribution weights), shape varies over time.
    tau : np.ndarray
        Time vector for each belief cohort (time since each cohort entered the economy), shape varies over time.
    """

    
    Npre = int(That / dt)  # Pre-trading period (converted to an integer for indexing)
    
    Zt = np.insert(np.cumsum(dZt), 0, 0)  # Brownian motion simulation
    yg = (mu_Y - 0.5 * sigma_Y**2) * dt + sigma_Y * dZt  # Log-growth rate process
    Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)  # Simulated output

    Xt = np.ones(Nt) 
    Deltabar = np.zeros(Nt)
    Delta_s_t = np.array([0.0])  # Initialized as a numpy array

    IntVec = np.array([nu * bet])  # Integration vector
    tau = np.array([dt])  # Time vector
    
    reduction = np.exp(-nu * dt)  # Discounting factor

    for i in range(1, Nt):
        Part = IntVec * np.exp(-(rho + 0.5 * Delta_s_t**2) * dt + Delta_s_t * dZt[i - 1])

        Xt[i] = np.sum(Part)
        Deltabar[i] = np.sum(Part * Delta_s_t) / Xt[i]
        
        IntVec = np.append(reduction * Part, bet * (1 - reduction) * Xt[i])  # Update integration vector
        f = IntVec / Xt[i]  # Normalize

        # Compute belief updates using the posterior variance function
        dDelta_s_t = (post_var(sigma_Y, Vbar, tau) / sigma_Y**2) * (-Delta_s_t * dt + dZt[i - 1])

        if i < (Npre + 1):
            Delta_s_t = np.append(Delta_s_t + dDelta_s_t, 0)
        else:
            DELbias = np.sum(dZt[i - Npre:i]) / That
            Delta_s_t = np.append(Delta_s_t + dDelta_s_t, DELbias)

        tau = np.append(tau + dt, 0)  # Update time vector

    return Deltabar, IntVec, Xt, Delta_s_t, Yt, Zt, f, tau
