This repository contains a Python implementation of the code used to replicate the results from
"Asset Prices and Portfolio Choice with Learning from Experience" by Ehling, Graniero, and Heyerdahl-Larsen (2017).

The repository contains the following Python modules:
main.py
Main file to replicate the full set of results from Section 3 in the paper, including Figure 1, Figure 2, Figure 3, and Table 1.

postvar.py
Function that calculates the posterior variance of the Kalman filter.

build_cohorts.py
Function to create the stationary economy with a large number of cohorts.

sim_cohorts.py
Function to simulate the economy forward.
The default number of simulated paths is set to 100.
Note: Results in the paper are based on 10,000 simulated paths for robustness.

single_simulation.py  
Contains the `simulate_single_path` function used to simulate a single path of the economy.  
This function is designed for parallel processing using `joblib` and is called from `main.py`.  
It returns key time series and belief dynamics used for figures and regression results.

olsgmm.py
Function to perform OLS regressions with GMM-corrected standard errors.
Used to obtain the regression results in Table 1.

save_plots.py
Utility function to save the figures automatically as PNG files.
This function ensures that if a figure already exists, it will be overwritten with the latest version.
All figures are saved in a figures/ folder created automatically in the directory.

How to run the script:

Step 1:
Install all dependencies via: pip install -r requirements.txt
NOTE: It is recommended to work within a virtual environment

Step 2:
In main.py, set parameters you want.
Note: This program is designed for parallel processing, and the default is to run on all avaliable cores (n_jobs = -1). Set this as a smaller number than max available cores on your machine. 
Run the main script: python src/main.py
