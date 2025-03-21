# Main code for the numerical resuls in "Asset Prices and Portfolio Choice with Learning from Experience" 
# By Paul Ehling, Alessandro Graniero, and Christian Heyerdahl-Larsen

import numpy as np
import matplotlib.pyplot as plt
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
rho = 0.001  # Time discount rate
nu = 0.02  # Death rate
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_S = sigma_Y  # In equilibrium, stock price diffusion equals output diffusion
w = 0.92  # Fraction of total output paid out as dividends

# Pre-calculations
D = rho**2 + 4 * (rho * nu + nu**2) * (1 - w)
bet = (rho + 2 * nu - np.sqrt(D)) / (2 * nu)
rlog = rho + mu_Y - sigma_Y**2

# Setting prior variance
That = 20  # Pre-trading period
dt = 1 / 12
Npre = int(That / dt)  # Convert to integer since array indexing requires integers
Vbar = (sigma_Y**2) / That  # Prior variance
Tcohort = 500  # Time horizon to keep track of cohorts
Nt = int(Tcohort / dt)  # Convert to integer to match array handling

# Build up cohorts
MC = 1  # Could draw multiple build-ups
fMAT = np.zeros((MC, Nt))

for i in range(MC):
    dZt = np.sqrt(dt) * np.random.randn(Nt - 1)  # Generate random shocks
    Deltabar, IntVec, Xt, Delta_s_t, Yt, Zt, f, tau = build_up_cohorts(
        dZt, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, bet, That
    )
    fMAT[i, :] = f  # Store results

# Initialize variables for simulation
Mpaths = 100  # Number of simulation paths
Tsample = Tcohort / 100
Nsamples = 100
stepCorr = int(Tsample * 12)  # Convert to integer

# Initialize matrices for storing results
corrZport = np.zeros((Mpaths, Nsamples))
corrZMUs_t = np.zeros((Mpaths, Nsamples))
corrMU_sMUs_t = np.zeros((Mpaths, Nsamples))
corrMuSmuHat = np.zeros((Mpaths, 1))

fMAT = np.zeros((Mpaths, Nt))
mC = np.zeros((Mpaths, Nt))
sC = np.zeros((Mpaths, Nt))
DeltaHatMAT = np.zeros((Mpaths, Nt))
rMAT = np.zeros((Mpaths, Nt))
thetaMAT = np.zeros((Mpaths, Nt))
portMAT = np.zeros((Mpaths, Nt))
Zmat = np.zeros((Mpaths, Nt))

# Expected returns
muSMAT = np.zeros((Mpaths, Nt))      # Expected returns under true measure
muSsMAT = np.zeros((Mpaths, Nt))     # Expected returns under the agent's measure
muShatMAT = np.zeros((Mpaths, Nt))   # Simple average of expected returns (consensus belief)

# Other economic variables
EtMAT = np.zeros((Mpaths, Nt))
VtMAT = np.zeros((Mpaths, Nt))
RxMAT = np.zeros((Mpaths, Nt))

# Matrices for sample statistics
muCst = np.zeros((Mpaths, Nsamples))
logmuCst = np.zeros((Mpaths, Nsamples))
sigCst = np.zeros((Mpaths, Nsamples))
stdCst = np.zeros((Mpaths, Nsamples))

print(f"Running {Mpaths} paths in parallel...")

# Change n_jobs as the preferred number of cores
results = Parallel(n_jobs=6, backend="loky", verbose=10, batch_size="auto")(
    delayed(simulate_single_path)(
        k, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, sigma_S, bet, That, Npre, tau, IntVec, Delta_s_t, rlog
    )
    for k in range(Mpaths)
)

# --- UNPACK THE RESULTS ---
for k, res in enumerate(results):
    (dR, Et, Vt, Deltabar2, r_t, theta_t, Zt_k, Port, mu_S_adj, mu_S_t_adj, muhat_S_t_adj,
     muC_s_t, sigmaC_s_t, f_avg, corr_muS_muHat) = res

    # Fill corresponding matrices
    RxMAT[k] = dR
    EtMAT[k, :] = Et
    VtMAT[k, :] = Vt
    DeltaHatMAT[k, :] = Deltabar2
    rMAT[k, :] = r_t
    thetaMAT[k, :] = theta_t
    Zmat[k, :] = Zt_k
    portMAT[k, :] = Port
    muSMAT[k, :] = mu_S_adj
    muSsMAT[k, :] = mu_S_t_adj
    muShatMAT[k, :] = muhat_S_t_adj
    mC[k, :] = muC_s_t
    sC[k, :] = sigmaC_s_t
    fMAT[k, :] = f_avg
    corrMuSmuHat[k] = corr_muS_muHat

# Matrices for sample statistics (computed AFTER collecting all raw series)
corrZport = np.zeros((Mpaths, Nsamples))
corrZMUs_t = np.zeros((Mpaths, Nsamples))
corrMU_sMUs_t = np.zeros((Mpaths, Nsamples))
muCst = np.zeros((Mpaths, Nsamples))
logmuCst = np.zeros((Mpaths, Nsamples))
sigCst = np.zeros((Mpaths, Nsamples))
stdCst = np.zeros((Mpaths, Nsamples))

# Now loop over paths and windows for correlations like before
for k in range(Mpaths):
    for l in range(Nsamples):
        start, end = l * stepCorr, (l + 1) * stepCorr
        corrZMUs_t[k, l] = np.corrcoef(Zmat[k, start:end], muSsMAT[k, start:end])[0, 1]
        corrZport[k, l] = np.corrcoef(Zmat[k, start:end], portMAT[k, start:end])[0, 1]
        corrMU_sMUs_t[k, l] = np.corrcoef(muSMAT[k, start:end], muSsMAT[k, start:end])[0, 1]
        muCst[k, l] = np.mean(mC[k, start:end])
        logmuCst[k, l] = np.mean(mC[k, start:end] - 0.5 * sC[k, start:end] ** 2)
        sigCst[k, l] = np.mean(sC[k, start:end])
        stdCst[k, l] = np.mean(np.abs(sC[k, start:end]))


# Define MaxAge and compute related parameters
MaxAge = 100
MaxAgeN = MaxAge // 5  
tperiod = np.arange(Tsample, 101, Tsample)  

# Compute mean values from simulations
meanZport = np.mean(corrZport, axis=0)  # Mean along paths
meanZmus_t = np.mean(corrZMUs_t, axis=0)

meanMus = np.mean(corrMU_sMUs_t, axis=0)
meanMuCst = np.mean(muCst, axis=0)
meanSCst = np.mean(sigCst, axis=0)
meanStdCst = np.mean(stdCst, axis=0)
meanLogMuCst = np.mean(logmuCst, axis=0)

# --- FIGURE 1 ---
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
axes[0].plot(tperiod, meanMus[:MaxAgeN])
axes[0].set_xlabel("Age")
axes[0].set_ylabel(r"corr( $\mu^S_t - r_t$, $\mu^S_{s,t} - r_t$ )")

# Second subplot
axes[1].plot(tperiod, meanZmus_t[:MaxAgeN])
axes[1].set_xlabel("Age")
axes[1].set_ylabel(r"corr( $z_t$, $\mu^S_t - r_t$ )")

# Save Figure 1
save_plot(fig1, "figure1.png")
plt.close(fig1)  # Close figure to free memory

# --- FIGURE 2 ---
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(tperiod, meanZport[:MaxAgeN])
ax2.set_xlabel("Age")
ax2.set_ylabel(r"corr( $z_t$, $\pi_{s,t}$ )")

# Save Figure 2
save_plot(fig2, "figure2.png")
plt.close(fig2)

# --- FIGURE 3 ---
fig3, axes3 = plt.subplots(2, 1, figsize=(6, 8))

# First subplot
axes3[0].plot(tperiod, meanLogMuCst[:MaxAgeN])
axes3[0].set_xlabel("Age")
axes3[0].set_ylabel("Drift of log consumption")

# Second subplot
axes3[1].plot(tperiod, meanStdCst[:MaxAgeN])
axes3[1].set_xlabel("Age")
axes3[1].set_ylabel("Volatility of log consumption growth")

# Save Figure 3
save_plot(fig3, "figure3.png")
plt.close(fig3)

# Compute mean values of market view diffusion
mean_market_diffusion = np.mean(VtMAT + EtMAT)
relative_importance = np.mean(EtMAT) / np.mean(VtMAT + EtMAT)
relative_contribution = np.mean(EtMAT / (VtMAT + EtMAT))

# Print results
print(f"Mean value of market view diffusion: {mean_market_diffusion:.6f}")
print(f"Relative importance: {relative_importance:.6f}")
print(f"Relative contribution: {relative_contribution:.6f}")

# Initialize arrays for standard deviations and correlations
stdRPtrue = np.zeros(Mpaths)
stdRPsurvey = np.zeros(Mpaths)
corrRPtruesurvey = np.zeros(Mpaths)

# Loop over each path to compute std and correlation
for k in range(Mpaths):
    mutrue = muSMAT[k, :]
    muSurvey = muShatMAT[k, :]
    stdRPtrue[k] = np.std(mutrue)  # Standard deviation of true risk premia
    stdRPsurvey[k] = np.std(muSurvey)  # Standard deviation of survey risk premia
    corrRPtruesurvey[k] = np.corrcoef(mutrue, muSurvey)[0, 1]  # Correlation between true and survey RP

# Compute mean values
mean_stdRPtrue = np.mean(stdRPtrue)
mean_stdRPsurvey = np.mean(stdRPsurvey)
mean_ratio_std = np.mean(stdRPsurvey / stdRPtrue)  # Jensen's term difference
simple_ratio_std = np.mean(stdRPsurvey) / np.mean(stdRPtrue)  # Standard ratio
mean_corrRPtruesurvey = np.mean(corrRPtruesurvey)

# Print results
print(f"Std of true RP: {mean_stdRPtrue:.6f}")
print(f"Std of survey RP: {mean_stdRPsurvey:.6f}")
print(f"Ratio of std survey/std true (Jensen's term): {mean_ratio_std:.6f}")
print(f"Simple ratio of std survey/std true: {simple_ratio_std:.6f}")
print(f"Corr RP true and RP survey: {mean_corrRPtruesurvey:.6f}")

# Initialize matrices for regression results
RegsExtrapSurvey = np.zeros((Mpaths, 5))
RegsExtrapTrue = np.zeros((Mpaths, 5))
RegsRPtrueSurvey = np.zeros((Mpaths, 5))

# Creating 12-month (overlapping) returns
RMAT12 = np.zeros((Mpaths, Nt - 12))

for j in range(Nt - 12):
    RMAT12[:, j] = np.sum(RxMAT[:, j + 1:j + 12] + rMAT[:, j:j + 11], axis=1)


# Running regressions for each path
for i in range(Mpaths):
    R12 = RMAT12[i, :-1]  # Equivalent to MATLAB's "R12 = RMAT12(i,1:end-1)'"
    mS = (muShatMAT[i, :] + nu * (1 - bet)) * dt * 12  # Survey mean
    mT = (muSMAT[i, :] + nu * (1 - bet)) * dt * 12  # True mean
    
    # Regression (1): True Expected Returns on Past 12-month Returns
    X = np.column_stack((np.ones(R12.shape[0]), R12))  # Add constant term
    X = X[:mT[14:].shape[0]]

    bv, sebv, R2v, R2vadj, v, F = olsgmm(mT[14:].reshape(-1, 1), X, 1, 1)
    t_values = bv / sebv

    RegsExtrapTrue[i, :2] = bv.flatten()  # Store coefficients
    RegsExtrapTrue[i, 2:4] = t_values.flatten()  # Store t-statistics
    RegsExtrapTrue[i, 4] = R2v.item()  # Store R² value

    # Regression (2): Survey Expected Returns on Past 12-month Returns
    bv, sebv, R2v, R2vadj, v, F = olsgmm(mS[14:].reshape(-1, 1), X, 1, 1)
    t_values = bv / sebv

    RegsExtrapSurvey[i, :2] = bv.flatten()
    RegsExtrapSurvey[i, 2:4] = t_values.flatten()
    RegsExtrapSurvey[i, 4] = R2v.item()

    # Regression (3): True Expected Returns on Survey Expected Returns
    X = np.column_stack((np.ones_like(mS), mS))  # Add constant term
    bv, sebv, R2v, R2vadj, v, F = olsgmm(mT.reshape(-1, 1), X, 1, 1)
    t_values = bv / sebv

    RegsRPtrueSurvey[i, :2] = bv.flatten()
    RegsRPtrueSurvey[i, 2:4] = t_values.flatten()
    RegsRPtrueSurvey[i, 4] = R2v.item()

# Print Table 1 results
print("\n### Table 1 Results ###")

# Regression (1): Survey Expected Returns on Past 12-Month Returns
print("\nReg (1) - beta:", np.mean(RegsExtrapSurvey[:, 1]))
print("Reg (1) - t-stat:", np.mean(RegsExtrapSurvey[:, 3]))
print("Reg (1) - R2:", np.mean(RegsExtrapSurvey[:, -1]))

# Regression (2): True Expected Returns on Past 12-Month Returns
print("\nReg (2) - beta:", np.mean(RegsExtrapTrue[:, 1]))
print("Reg (2) - t-stat:", np.mean(RegsExtrapTrue[:, 3]))
print("Reg (2) - R2:", np.mean(RegsExtrapTrue[:, -1]))

# Regression (3): True Expected Returns on Survey Expected Returns
print("\nReg (3) - beta:", np.mean(RegsRPtrueSurvey[:, 1]))
print("Reg (3) - t-stat:", np.mean(RegsRPtrueSurvey[:, 3]))
print("Reg (3) - R2:", np.mean(RegsRPtrueSurvey[:, -1]))

toc()