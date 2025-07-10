Master Thesis KB McDonnell with Marianne Bauer Group sept 2024 - juli 2025, TU Delft

Welcome to the Github for my thesis project. 

This github contains all my coding work done for this project. The main focus of the project ended up being the Delta-Hes model. 
But briefly: the phase model folder contains a delayed kuramoto-like phase model, the data analysis is done for data on single cell Hes oscillations of 

Delta-Hes folder is structured as follows: 

It contains 3 python .py files. These contain all the functions used in the project. 
delta_hes_model.py: contains all simulation functions
analysis.py: contains all functions to obtain observables from the simulation functions
visualisation.py: contains tools to visualise the data with particular colormaps or to make animations

Then there are a number of Jupyter Notebook .ipynb files that run the simulations, calculate observables and plot results for the various sub-models discussed in the report
internal_oscillator.ipynb: runs all code for the internal oscillator
external_delta_source.ipynb: runs all code for the external delta source
two_coupled_cells.ipynb: runs all code for the two coupled cells system
1D_coupled_cells.ipynb: runs all code for the 1D array of coupled cells system 

Then there are two Jupyter Notebook files that animate different systems: for a 1D system (simulate_1D.ipynb) and a 2D system (simulate_2D.ipynb)

comparison-two-cell.npz: is a data file which contains information on the period of the two cell case to be able to compare periods of the 1D case in 1D_coupled_cells.ipynb
report_plotting_misc.ipynb: contains a couple of plots for the report
