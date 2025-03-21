import numpy as np
from postvar import post_var  # Import posterior variance function

def sim_cohorts(biasvec: np.ndarray, dZt: np.ndarray, Nt: int, tau: np.ndarray,
                IntVec: np.ndarray, Delta_s_t: np.ndarray, dt: float, rho: float, nu: float,
                Vbar: float, mu_Y: float, sigma_Y: float, sigma_S: float, bet: float,
                That: float, Npre: int):
    """
    Simulates the forward dynamics of the economy with overlapping cohorts under 
    a learning-from-experience framework.

    This function propagates agents’ beliefs and economic variables forward in time 
    given initial conditions and stochastic shocks, tracking the evolution of beliefs, 
    expected returns, consumption dynamics, and other macro-financial variables across cohorts.

    Parameters:
    -----------
    biasvec : np.ndarray
        Vector containing the initial bias term used for belief correction, length Npre.
    dZt : np.ndarray
        Brownian motion increments (shocks), shape (Nt,).
    Nt : int
        Total number of simulation time steps.
    tau : np.ndarray
        Vector of cohort ages (time since entry).
    IntVec : np.ndarray
        Initial integration vector for cohort weights (based on past beliefs).
    Delta_s_t : np.ndarray
        Initial vector of cohort-specific beliefs about the state variable (e.g., drift).
    dt : float
        Time step size (e.g., 1/12 for monthly data).
    rho : float
        Time preference rate.
    nu : float
        Death or cohort replacement rate.
    Vbar : float
        Prior belief variance about the state variable.
    mu_Y : float
        Drift (mean growth) of the output process.
    sigma_Y : float
        Volatility of the output process.
    sigma_S : float
        Volatility of the asset return process.
    bet : float
        Steady-state consumption share parameter.
    That : float
        Duration of the pre-simulation (build-up) period in years.
    Npre : int
        Number of periods in the pre-trading phase.

    Returns:
    --------
    Xt2 : np.ndarray
        Total weight (normalization constant) on beliefs at each time, shape (Nt,).
    Deltabar2 : np.ndarray
        Weighted average belief across cohorts at each time, shape (Nt,).
    part1 : np.ndarray
        Experience-based learning term (moving average of shocks), shape (Nt,).
    mu_S : np.ndarray
        Expected excess return under the true measure, shape (Nt,).
    mu_S_t : np.ndarray
        Expected excess return under the subjective measure, shape (Nt,).
    muhat_S_t : np.ndarray
        Average belief across cohorts about expected returns, shape (Nt,).
    r_t : np.ndarray
        Risk-free rate at each time, shape (Nt,).
    theta_t : np.ndarray
        Price of risk (Sharpe ratio) at each time, shape (Nt,).
    Port : np.ndarray
        Optimal portfolio choice at each time step, shape (Nt,).
    muC_s_t : np.ndarray
        Drift of log consumption growth across cohorts, shape (Nt,).
    sigmaC_s_t : np.ndarray
        Volatility of log consumption growth across cohorts, shape (Nt,).
    BIGf : np.ndarray
        Matrix of normalized cohort weights over time, shape (Nt, Nt).
    BIGDELTA : np.ndarray
        Matrix of cohort beliefs over time, shape (Nt, Nt).
    Et : np.ndarray
        Cross-sectional expectation (mean) of belief deviations, shape (Nt,).
    Vt : np.ndarray
        Cross-sectional variance of belief deviations, shape (Nt,).
    dR : np.ndarray
        Asset returns including dividends, shape (Nt,).
    """

    # Precompute posterior variance based on tau (cohort specific)
    post_var_vec = post_var(sigma_Y, Vbar, tau)

    # --- Preallocating arrays ---
    Xt2 = np.zeros(Nt)
    Deltabar2 = np.zeros(Nt)
    Et = np.zeros(Nt)
    Vt = np.zeros(Nt)
    part1 = np.zeros(Nt)
    dR = np.zeros(Nt)
    BIGDELTA = np.zeros((Nt, Nt))
    BIGf = np.zeros((Nt, Nt))

    # Reverse weights
    reduction = np.exp(-nu * dt)
    RevNt = np.arange(Nt, 0, -1)
    fhat = (reduction ** (RevNt - 1)) * nu * dt
    fhat /= np.sum(fhat)  # Normalize fhat

    # Other preallocated arrays
    mu_S = np.zeros(Nt)
    mu_S_t = np.zeros(Nt)
    muhat_S_t = np.zeros(Nt)
    muC_s_t = np.zeros(Nt)
    sigmaC_s_t = np.zeros(Nt)
    r_t = np.zeros(Nt)
    theta_t = np.zeros(Nt)
    fst = np.zeros(Nt)

    bias_weight = (Vbar / (1 + (Vbar / sigma_Y**2) * dt * RevNt)) * (1 / sigma_Y)

    # Loop over time steps
    for i in range(Nt):
        # Update integral vector and normalized weights
        Part = IntVec * np.exp(-0.5 * Delta_s_t**2 * dt + Delta_s_t * dZt[i])
        Xt2[i] = np.sum(Part)
        Deltabar2[i] = np.sum(Part * Delta_s_t) / Xt2[i]
        f = Part / Xt2[i]

        # Belief updating for part1 (experience term)
        if i == 0:
            part1[i] = np.sum(biasvec) / That
        else:
            part1[i] = part1[i - 1] + (post_var(sigma_Y, Vbar, i * dt) / sigma_Y**2) * (-part1[i - 1] * dt + dZt[i - 1])

        # Dividend return
        dR[i] = (mu_S[i - 1] - r_t[i - 1] + rho + mu_Y - sigma_Y**2 + nu * (1 - bet)) * dt + sigma_S * dZt[i]

        # Store current Delta and f
        BIGDELTA[i, :] = Delta_s_t
        BIGf[i, :] = f

        # Compute expected returns
        mu_S[i] = sigma_S * sigma_Y - (sigma_S - sigma_Y) * Deltabar2[i]
        mu_S_t[i] = sigma_S * sigma_Y + sigma_S * (-Deltabar2[i] + part1[i]) + sigma_Y * Deltabar2[i]
        muhat_S_t[i] = mu_S[i] + sigma_S * fhat @ Delta_s_t

        # Risk-free rate and price of risk
        r_t[i] = rho + mu_Y - sigma_Y**2 + nu * (1 - bet) + sigma_Y * Deltabar2[i]
        theta_t[i] = sigma_Y - Deltabar2[i]

        # Expectation and variance of beliefs
        Et[i] = f @ bias_weight
        Vt[i] = (f @ Delta_s_t**2) * sigma_Y - Deltabar2[i]**2 * sigma_Y

        fst[i] = f[Nt - (i + 1)]  # Aligning weight for cohort

        # Consumption drift and volatility
        muC_s_t[i] = mu_Y + nu * (1 - bet) + (sigma_Y - Deltabar2[i]) * (part1[i] - Deltabar2[i])
        sigmaC_s_t[i] = sigma_Y + part1[i] - Deltabar2[i]

        # --- Belief deviations update (optimized using precomputed post_var_vec) ---
        dDelta_s_t = (post_var_vec / sigma_Y**2) * (-Delta_s_t * dt + dZt[i])

        # Bias adjustment depending on i
        if i < Npre:
            DELbias = (np.sum(biasvec[i:]) + np.sum(dZt[:i])) / That
        else:
            DELbias = np.sum(dZt[i - Npre + 1: i + 1]) / That  # Corrected indexing!

        # Update Delta_s_t and IntVec efficiently
        Delta_s_t = np.append(Delta_s_t[1:] + dDelta_s_t[1:], DELbias)
        IntVec = np.append(reduction * Part[1:], bet * (1 - reduction) * Xt2[i])

    # Portfolio choice (final computation)
    Port = (part1 - Deltabar2) / sigma_S + (sigma_Y / sigma_S) * (1 - bet * np.flip(fhat) / fst)

    return Xt2, Deltabar2, part1, mu_S, mu_S_t, muhat_S_t, r_t, theta_t, Port, muC_s_t, sigmaC_s_t, BIGf, BIGDELTA, Et, Vt, dR
