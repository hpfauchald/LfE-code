# Main code for the numerical results in "Asset Prices and Portfolio Choice with Learning from Experience"
# By Paul Ehling, Alessandro Graniero, and Christian Heyerdahl-Larsen

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from build_cohorts import build_up_cohorts
from olsgmm import olsgmm
from save_plots import save_plot
from joblib import Parallel, delayed
from single_simulation import simulate_single_path

# Define tic and toc globally
import time

def tic():
    global start_time
    start_time = time.perf_counter()

def toc():
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")


tic()

# Parameters
rho = 0.001
nu = 0.02
mu_Y = 0.02
sigma_Y = 0.033
sigma_S = sigma_Y
w = 0.92
n_jobs = 6

# Pre-calculations
# From the equilibrium model (Equation 2.12):
D = rho**2 + 4 * (rho * nu + nu**2) * (1 - w)
bet = (rho + 2 * nu - np.sqrt(D)) / (2 * nu)
rlog = rho + mu_Y - sigma_Y**2

# Setting prior variance
build_up_period = 20
dt = 1 / 12
n_pre_periods = int(build_up_period / dt)
prior_variance = (sigma_Y**2) / build_up_period
cohort_horizon = 500
n_timesteps = int(cohort_horizon / dt)

# Build up cohorts
n_build_paths = 1
belief_weight_matrix = np.zeros((n_build_paths, n_timesteps))

for i in range(n_build_paths):
    dz_t_init = np.sqrt(dt) * np.random.randn(n_timesteps - 1)
    (
        init_avg_belief,
        integration_vector,
        Xt,
        cohort_beliefs,
        Yt,
        Zt,
        f,
        tau,
    ) = build_up_cohorts(
        dz_t_init, n_timesteps, dt, rho, nu, prior_variance, mu_Y, sigma_Y, bet, build_up_period
    )
    belief_weight_matrix[i, :] = f

# Initialize variables for simulation
n_sim_paths = 100
sample_years = cohort_horizon / 100
n_samples = 100
sample_step = int(sample_years * 12)

# Initialize result containers
# These correspond to values used in Figure 1 and Figure 2
corr_z_portfolio = np.zeros((n_sim_paths, n_samples))
corr_z_subjective_return = np.zeros((n_sim_paths, n_samples))
corr_true_vs_subjective_return = np.zeros((n_sim_paths, n_samples))
corr_true_vs_consensus_return = np.zeros((n_sim_paths, 1))

# Initialize matrices for storing time series from each simulation path
# These track cohort beliefs, portfolio allocations, and macro variables
belief_weight_matrix = np.zeros((n_sim_paths, n_timesteps))
muC_matrix = np.zeros((n_sim_paths, n_timesteps))
sigmaC_matrix = np.zeros((n_sim_paths, n_timesteps))
avg_belief_matrix = np.zeros((n_sim_paths, n_timesteps))
r_matrix = np.zeros((n_sim_paths, n_timesteps))
theta_matrix = np.zeros((n_sim_paths, n_timesteps))
portfolio_matrix = np.zeros((n_sim_paths, n_timesteps))
Z_matrix = np.zeros((n_sim_paths, n_timesteps))

# Expected return matrices (used in regressions and statistics)
true_return_matrix = np.zeros((n_sim_paths, n_timesteps))
subjective_return_matrix = np.zeros((n_sim_paths, n_timesteps))
consensus_return_matrix = np.zeros((n_sim_paths, n_timesteps))

# Market belief dispersion (Eq. 2.12)
Et_matrix = np.zeros((n_sim_paths, n_timesteps))
Vt_matrix = np.zeros((n_sim_paths, n_timesteps))
return_matrix = np.zeros((n_sim_paths, n_timesteps))

# Sample-level stats
muC_sample = np.zeros((n_sim_paths, n_samples))
logmuC_sample = np.zeros((n_sim_paths, n_samples))
sigmaC_sample = np.zeros((n_sim_paths, n_samples))
stdC_sample = np.zeros((n_sim_paths, n_samples))

print(f"Running {n_sim_paths} paths in parallel...")

# Run n_sim_paths simulations in parallel using the simulate_single_path function
# Each simulation produces one full time series of variables like returns, beliefs, etc.
# This step reflects the stochastic simulation of the full model described in Sections 2.1–2.4
results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")(
    delayed(simulate_single_path)(
        k,
        n_timesteps,
        dt,
        rho,
        nu,
        prior_variance,
        mu_Y,
        sigma_Y,
        sigma_S,
        bet,
        build_up_period,
        n_pre_periods,
        tau,
        integration_vector,
        cohort_beliefs,
        rlog,
    )
    for k in tqdm(range(n_sim_paths), desc="Simulating Paths")
)

# Unpack simulation results
for k, res in enumerate(results):
    (
        dR,
        Et,
        Vt,
        avg_belief_series,
        r_t,
        theta_t,
        Zt_k,
        portfolio,
        mu_S_adj,
        mu_S_t_adj,
        muhat_S_t_adj,
        muC_s_t,
        sigmaC_s_t,
        f_avg,
        corr_muS_muHat,
    ) = res

    # Store simulation results path-by-path into large matrices
    # These matrices will later be used to compute empirical moments, correlations, and figures
    return_matrix[k] = dR                            # Realized excess returns
    Et_matrix[k, :] = Et                             # Market-level learning variance (belief heterogeneity)
    Vt_matrix[k, :] = Vt                             # Subjective belief variance
    avg_belief_matrix[k, :] = avg_belief_series      # Cohort-averaged belief Δ̄ₜ
    r_matrix[k, :] = r_t                             # Risk-free rate
    theta_matrix[k, :] = theta_t                     # Sharpe ratio (price of risk)
    Z_matrix[k, :] = Zt_k                             # Cumulative Brownian motion path z_t
    portfolio_matrix[k, :] = portfolio               # Optimal portfolio allocation πₛ,ₜ

    true_return_matrix[k, :] = mu_S_adj              # Adjusted expected return under true measure
    subjective_return_matrix[k, :] = mu_S_t_adj      # Adjusted expected return under subjective beliefs
    consensus_return_matrix[k, :] = muhat_S_t_adj    # Cross-sectional expected return μ̂^S_t

    muC_matrix[k, :] = muC_s_t                       # Drift of log consumption
    sigmaC_matrix[k, :] = sigmaC_s_t                 # Volatility of log consumption
    belief_weight_matrix[k, :] = f_avg               # Final cohort weights f_τ,t
    corr_true_vs_consensus_return[k] = corr_muS_muHat  # Correlation of μ^S_t and μ̂^S_t


# Sample stats loop
for k in range(n_sim_paths):
    for l in range(n_samples):
        start, end = l * sample_step, (l + 1) * sample_step
        corr_z_subjective_return[k, l] = np.corrcoef(
            Z_matrix[k, start:end], subjective_return_matrix[k, start:end]
        )[0, 1]
        corr_z_portfolio[k, l] = np.corrcoef(
            Z_matrix[k, start:end], portfolio_matrix[k, start:end]
        )[0, 1]
        corr_true_vs_subjective_return[k, l] = np.corrcoef(
            true_return_matrix[k, start:end], subjective_return_matrix[k, start:end]
        )[0, 1]
        muC_sample[k, l] = np.mean(muC_matrix[k, start:end])
        logmuC_sample[k, l] = np.mean(muC_matrix[k, start:end] - 0.5 * sigmaC_matrix[k, start:end] ** 2)
        sigmaC_sample[k, l] = np.mean(sigmaC_matrix[k, start:end])
        stdC_sample[k, l] = np.mean(np.abs(sigmaC_matrix[k, start:end]))


# Define MaxAge and compute related parameters
max_age = 100
max_age_steps = max_age // 5  
time_grid = np.arange(sample_years, 101, sample_years)

# Compute mean values from simulations
mean_corr_z_portfolio = np.mean(corr_z_portfolio, axis=0)
mean_corr_z_subjective_return = np.mean(corr_z_subjective_return, axis=0)
mean_corr_true_vs_subjective = np.mean(corr_true_vs_subjective_return, axis=0)
mean_muC = np.mean(muC_sample, axis=0)
mean_sigmaC = np.mean(sigmaC_sample, axis=0)
mean_stdC = np.mean(stdC_sample, axis=0)
mean_log_muC = np.mean(logmuC_sample, axis=0)

# --- FIGURE 1 ---
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
axes[0].plot(time_grid, mean_corr_true_vs_subjective[:max_age_steps])
axes[0].set_xlabel("Age")
axes[0].set_ylabel(r"corr( $\mu^S_t - r_t$, $\mu^S_{s,t} - r_t$ )")

# Second subplot
axes[1].plot(time_grid, mean_corr_z_subjective_return[:max_age_steps])
axes[1].set_xlabel("Age")
axes[1].set_ylabel(r"corr( $z_t$, $\mu^S_t - r_t$ )")

# Save Figure 1
save_plot(fig1, "figure1.png")
plt.close(fig1)

# --- FIGURE 2 ---
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(time_grid, mean_corr_z_portfolio[:max_age_steps])
ax2.set_xlabel("Age")
ax2.set_ylabel(r"corr( $z_t$, $\pi_{s,t}$ )")

# Save Figure 2
save_plot(fig2, "figure2.png")
plt.close(fig2)

# --- FIGURE 3 ---
fig3, axes3 = plt.subplots(2, 1, figsize=(6, 8))

# First subplot
axes3[0].plot(time_grid, mean_log_muC[:max_age_steps])
axes3[0].set_xlabel("Age")
axes3[0].set_ylabel("Drift of log consumption")

# Second subplot
axes3[1].plot(time_grid, mean_stdC[:max_age_steps])
axes3[1].set_xlabel("Age")
axes3[1].set_ylabel("Volatility of log consumption growth")

# Save Figure 3
save_plot(fig3, "figure3.png")
plt.close(fig3)

# Compute mean values of market view diffusion
mean_market_diffusion = np.mean(Vt_matrix + Et_matrix)
relative_importance = np.mean(Et_matrix) / np.mean(Vt_matrix + Et_matrix)
relative_contribution = np.mean(Et_matrix / (Vt_matrix + Et_matrix))

# Print results
print(f"Mean value of market view diffusion: {mean_market_diffusion:.6f}")
print(f"Relative importance: {relative_importance:.6f}")
print(f"Relative contribution: {relative_contribution:.6f}")

# Initialize arrays for standard deviations and correlations
std_true_rp = np.zeros(n_sim_paths)
std_survey_rp = np.zeros(n_sim_paths)
corr_true_vs_survey_rp = np.zeros(n_sim_paths)

# Loop over each path to compute std and correlation
for k in range(n_sim_paths):
    true_rp = true_return_matrix[k, :]
    survey_rp = consensus_return_matrix[k, :]
    std_true_rp[k] = np.std(true_rp)
    std_survey_rp[k] = np.std(survey_rp)
    corr_true_vs_survey_rp[k] = np.corrcoef(true_rp, survey_rp)[0, 1]

# Compute mean values
mean_std_true_rp = np.mean(std_true_rp)
mean_std_survey_rp = np.mean(std_survey_rp)
mean_ratio_std = np.mean(std_survey_rp / std_true_rp)
simple_ratio_std = np.mean(std_survey_rp) / np.mean(std_true_rp)
mean_corr_true_vs_survey_rp = np.mean(corr_true_vs_survey_rp)

# Print results
print(f"Std of true RP: {mean_std_true_rp:.6f}")
print(f"Std of survey RP: {mean_std_survey_rp:.6f}")
print(f"Ratio of std survey/std true (Jensen's term): {mean_ratio_std:.6f}")
print(f"Simple ratio of std survey/std true: {simple_ratio_std:.6f}")
print(f"Corr RP true and RP survey: {mean_corr_true_vs_survey_rp:.6f}")

# Initialize matrices for regression results
regs_extrap_survey = np.zeros((n_sim_paths, 5))
regs_extrap_true = np.zeros((n_sim_paths, 5))
regs_true_vs_survey = np.zeros((n_sim_paths, 5))

# Creating 12-month (overlapping) returns
rmatrix_12mo = np.zeros((n_sim_paths, n_timesteps - 12))

for j in range(n_timesteps - 12):
    rmatrix_12mo[:, j] = np.sum(
        return_matrix[:, j + 1 : j + 12] + r_matrix[:, j : j + 11], axis=1
    )

# Running regressions for each path
for i in range(n_sim_paths):
    R12 = rmatrix_12mo[i, :-1]
    survey_mean = (consensus_return_matrix[i, :] + nu * (1 - bet)) * dt * 12
    true_mean = (true_return_matrix[i, :] + nu * (1 - bet)) * dt * 12

    X = np.column_stack((np.ones(R12.shape[0]), R12))
    X = X[: true_mean[14:].shape[0]]

    bv, sebv, R2v, R2vadj, v, F = olsgmm(true_mean[14:].reshape(-1, 1), X, 1, 1)
    t_values = bv / sebv
    regs_extrap_true[i, :2] = bv.flatten()
    regs_extrap_true[i, 2:4] = t_values.flatten()
    regs_extrap_true[i, 4] = R2v.item()

    bv, sebv, R2v, R2vadj, v, F = olsgmm(survey_mean[14:].reshape(-1, 1), X, 1, 1)
    t_values = bv / sebv
    regs_extrap_survey[i, :2] = bv.flatten()
    regs_extrap_survey[i, 2:4] = t_values.flatten()
    regs_extrap_survey[i, 4] = R2v.item()

    X = np.column_stack((np.ones_like(survey_mean), survey_mean))
    bv, sebv, R2v, R2vadj, v, F = olsgmm(true_mean.reshape(-1, 1), X, 1, 1)
    t_values = bv / sebv
    regs_true_vs_survey[i, :2] = bv.flatten()
    regs_true_vs_survey[i, 2:4] = t_values.flatten()
    regs_true_vs_survey[i, 4] = R2v.item()

# Print Table 1 results
print("\n### Table 1 Results ###")

# Regression (1)
print("\nReg (1) - beta:", np.mean(regs_extrap_survey[:, 1]))
print("Reg (1) - t-stat:", np.mean(regs_extrap_survey[:, 3]))
print("Reg (1) - R2:", np.mean(regs_extrap_survey[:, -1]))

# Regression (2)
print("\nReg (2) - beta:", np.mean(regs_extrap_true[:, 1]))
print("Reg (2) - t-stat:", np.mean(regs_extrap_true[:, 3]))
print("Reg (2) - R2:", np.mean(regs_extrap_true[:, -1]))

# Regression (3)
print("\nReg (3) - beta:", np.mean(regs_true_vs_survey[:, 1]))
print("Reg (3) - t-stat:", np.mean(regs_true_vs_survey[:, 3]))
print("Reg (3) - R2:", np.mean(regs_true_vs_survey[:, -1]))

toc()
