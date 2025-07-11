"""This module provides functions to obtain observables from time signals, such as estimating periods and amplitudes from peaks, finding phase shifts, and calculating decay ratios. 
It also includes functions for analysing simulation results of two-cell systems, for different initialisations and an inclusion of delay."""

import numpy as np
from scipy.signal import find_peaks

# functions to obtain observables from a time signal
def estimate_period_from_peaks(signal, time=None, height=None, distance=None, prominence=None):
    """
    Estimate the period of an oscillating signal using peak detection.

    Parameters:
    - signal: 1D numpy array of the signal values
    - time: Optional 1D array of time values (same length as signal). If None, assume uniform time steps.
    - height, distance, prominence: Optional arguments passed to find_peaks

    Returns:
    - period: Estimated average period of the oscillation, if enough peaks are found; otherwise None.
    - peak_times: Time values of the detected peaks in mins
    """
    # Find peaks
    peaks, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        return None, np.array([])  # Not enough peaks to estimate period

    # Assume uniform spacing if time is not given
    if time is None:
        time = np.arange(len(signal))

    peak_times = time[peaks]

    if len(peak_times) < 2:
        return None, peak_times  # Still not enough peaks

    # Calculate differences between consecutive peaks
    peak_diffs = np.diff(peak_times)
    period = np.mean(peak_diffs)

    return period, peak_times

def estimate_amplitude_from_peaks(signal, height=None, distance=None, prominence=None):
    """
    Estimate the amplitude of an oscillating signal using peak and trough detection.

    Parameters:
    - signal: 1D numpy array of the signal values
    - height, distance, prominence: Optional arguments passed to find_peaks

    Returns:
    - amplitude: Estimated average amplitude (float)
    - peak_indices: Indices of the used peaks
    - trough_indices: Indices of the used troughs
    """
    peak_indices, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)
    trough_indices, _ = find_peaks(-signal, height=height, distance=distance, prominence=prominence)

    if len(peak_indices) < 1 or len(trough_indices) < 1:
        return None, peak_indices, trough_indices

    peak_vals = signal[peak_indices]
    trough_vals = signal[trough_indices]

    min_len = min(len(peak_vals), len(trough_vals))
    if min_len < 1:
        return None, peak_indices, trough_indices

    amplitude = 0.5 * np.mean(peak_vals[:min_len] - trough_vals[:min_len])

    return amplitude, peak_indices[:min_len], trough_indices[:min_len]

def find_shift(variable1, variable2):
    """Find the phase shift in time between two variables by finding the closest peak value of variable 1 to each value in variable 2.
      If two peaks of variable 1 are both set on the same peak for variable 2, the one with the smallest distance is kept and the others are set to NaN.
      
      Inputs:
      - variable1: 1D numpy array of the first variable (e.g., peaks of h0)
      - variable2: 1D numpy array of the second variable (e.g., peaks of h1)
      
      Returns:
      - shift: The average time shift in seconds between the two variables, calculated as the mean of the distances to the closest peaks."""
    dt = 0.2

    variable1 = np.array(variable1)
    variable2 = np.array(variable2)

    # set up array to store the closest peak index and the distance to the closest peak
    closest = np.zeros(len(variable2))
    distances = np.zeros(len(variable2))

    if len(variable1) == 0 or len(variable2) == 0:
        return np.nan
    
    # Loop through each value in variable2 and find the closest value in variable1
    for k in range(len(variable2)):
        distance = variable2[k] - variable1
        distance[distance < 0] = np.inf  # Ignore negative distances

        # Find the index and value of the closest value in variable1
        closest[k] = np.argmin(distance)
        distances[k] = np.min(distance)

    # Filter out any duplicate closest peaks 
    unique, counts = np.unique(closest, return_counts=True)
    duplicates = unique[counts > 1]

    # Get the positions (indexes) of duplicates
    duplicate_positions = {val: np.where(closest == val)[0] for val in duplicates}

    # Set others that are not the closest to NaN (or -1 if you prefer an int flag)
    cleaned_closest = closest.astype(float)  # allow NaN
    for dup_val, positions in duplicate_positions.items():
        # Find the one with the smallest distance
        best_idx = positions[np.argmin(distances[positions])]
        # Set others to NaN
        for idx in positions:
            if idx != best_idx:
                cleaned_closest[idx] = np.nan
                distances[idx] = np.nan

    # Calculate the shift in time by taking the mean of the distances
    shift = np.nanmean(distances)*dt

    return shift

def mean_diff(result_diff, final_half = 8000):
    """ Calculate the mean and standard deviation of the differences between two variables over the final part of the time steps. 
    
    Inputs:
    - result_diff: A list containing two arrays, the first one is h_diff and the second one is d_diff, previously calculated as the absolute difference between two variables (e.g., h0 and h1, d0 and d1).
    - final_half: The number of time steps to consider from the end of the time series
    
    Returns:
    - mean_h_diff: Mean of the h_diff values over the final part of the time steps
    - std_h_diff: Standard deviation of the h_diff values over the final part of the time steps
    - mean_d_diff: Mean of the d_diff values over the final part of the time steps
    - std_d_diff: Standard deviation of the d_diff values over the final part of the time steps
    """

    h_diff = result_diff[0]
    d_diff = result_diff[1]

    mean_h_diff = np.mean(h_diff[:,:,final_half:], axis=2)
    std_h_diff = np.std(h_diff[:,:,final_half:], axis=2)

    mean_d_diff = np.mean(d_diff[:,:,final_half:], axis=2)
    std_d_diff = np.std(d_diff[:,:,final_half:], axis=2)

    return mean_h_diff, std_h_diff, mean_d_diff, std_d_diff

def decay_ratio(signal, time_frame=360, height=None, distance=None, prominence=None):
    """ Calculate the decay ratio of a time signal, by dividing two peak values that are separated by a minimum time frame. 

    Parameters:
    - signal: 1D numpy array of the time signal values.
    - time_frame: Minimum time frame in seconds between the two peaks to consider for the decay ratio.
    - height, distance, prominence: Optional arguments passed to find_peaks.

    Returns: 
    - decay_ratio: The ratio of the first peak value to the second peak value that is at least time_frame seconds away, or 100 if no such peak exists.
    """
    dt = 0.2  # Time step

    time_frame_steps = int(time_frame / dt)

    # Find peaks
    peaks, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)
    
    if len(peaks) < 2:
        return 100  # Not enough peaks

    # Calculate distances from the first peak
    peak_distances = peaks - peaks[0]
    mask = peak_distances > time_frame_steps

    if not np.any(mask):
        return 100  # No second peak far enough away

    # Get the first peak that satisfies the condition
    index = np.argmax(mask)

    peak_start = peaks[0]
    peak_end = peaks[index]

    peak_values_start = signal[peak_start]
    peak_values_end = signal[peak_end]

    # Check for invalid division
    if peak_values_end <= 0:
        return 100
    
    return peak_values_start / peak_values_end

def compute_and_plot_fft_spectrogram(signal, fs, window_size, cutoff_freq=None, dc_removal_start_idx=10000):
    """
    Computes FFT spectra for sliding windows and plots:
    1. Spectral magnitudes (or power) as an image.
    2. Signal power per window.

    Parameters:
    - signal: 1D numpy array, the time-domain signal.
    - fs: Sampling rate in Hz.
    - window_size: Number of samples in each window.
    - normalize: Whether to normalize each FFT vector.
    - db_scale: Whether to convert FFT magnitudes to decibel scale.
    - cutoff_freq: Maximum frequency (Hz) to include in plot and output.
    - dc_removal_start_idx: Index to begin computing mean for DC removal.
    """

    # --- Remove DC component from signal
    signal_mean = np.mean(signal[dc_removal_start_idx:])
    signal = signal - signal_mean

    step = 1  # slide window by 1 sample
    num_windows = len(signal) - window_size + 1
    fft_matrix = []
    power_per_window = []

    # Precompute frequency vector
    freqs = np.fft.fftfreq(window_size, d=1/fs)[:window_size // 2]

    # Apply cutoff mask if specified
    if cutoff_freq is not None:
        freq_mask = freqs <= cutoff_freq
        freqs = freqs[freq_mask]
    else:
        freq_mask = slice(None)  # keep all

    for i in range(num_windows):
        window = signal[i:i+window_size]
        fft_vals = np.abs(fft(window))[:window_size // 2]
        fft_vals = fft_vals[freq_mask]
        power = np.sum(window**2)

        fft_matrix.append(fft_vals)
        power_per_window.append(power)

    fft_matrix = np.array(fft_matrix).T  # shape: [frequencies x time]
    power_per_window = np.array(power_per_window)
    

    return fft_matrix, freqs, power_per_window

def freqs_differences(fft_matrix, power):
    """Calculate the differences in frequencies and power spectrum between the last time step and all previous time steps.
     
      Inputs:
      - fft_matrix: 2D numpy array of shape (num_frequencies, num_time_steps) containing the FFT magnitudes.
      - power: 1D numpy array of shape (num_time_steps,) containing the power spectrum for each time step.
      
      Returns:
      - freq_diff: 1D numpy array of shape (num_time_steps,) containing the differences in frequencies.
      - power_diff: 1D numpy array of shape (num_time_steps - 1,) containing the differences in power spectrum."""
    fft_matrix_normalised = fft_matrix / np.linalg.norm(fft_matrix, axis=0, keepdims=True)
    
    freq_diff = np.sum((fft_matrix_normalised - fft_matrix_normalised[:,-1][:,np.newaxis])**2, axis = 0)
    
    power_diff = (power[1:] - power[-1])**2
    return freq_diff, power_diff

#######################################################################################################

# functions to calculate all observables for a collection of simulations

def analysis_2cells(results, time_settled, parameter1, parameter2):
    """Analyse the results of a two-cell simulations (shape parameter1 x variable 2 x num_tsteps x num_cells x 2) to extract observables such as period, amplitude, and mean values for a seperation 
    of the cell into synchronised and laterally inhibited (LI) parts. This was used for two-cell simulations with a uniform initialisation
    
    Inputs:
    - results: 5D numpy array of shape (parameter1, parameter2, num_tsteps, num_cells, 2) containing the simulation results
    - time_settled: Number of time steps to ignore at the beginning of the simulation to allow the system to settle
    - parameter1: 1D numpy array of the first parameter values
    - parameter2: 1D numpy array of the second parameter values
    
    Returns:
    - result_diff: List containing the absolute differences between the two cells for h and d for each time step and parameter combination
    - result_diff_LI_mean: List containing the mean of the absolute differences between the two cells for h and d for the LI part for each parameter combination
    - result_LI_mean: List containing the mean values of h0, h1, d0, and d1 for the LI part for each parameter combination
    - result_synced_index: List containing the first and last index of the synchronised part of the results for each parameter combination
    - result_period_synced: List containing the period of the synchronised part of h0, h1, d0, and d1 for each parameter combination
    - result_amplitude_synced: List containing the amplitude of the synchronised part of h0, h1, d0, and d1 for each parameter combination
    - result_period_LI: List containing the period of the LI part of h0, h1, d0, and d1 for each parameter combination
    - result_amplitude_LI: List containing the amplitude of the LI part of h0, h1, d0, and d1 for each parameter combination
    """

    dt = 0.2

    # seperate the results into the different variables
    h0 = results[:,:,  time_settled:, 0, 0]
    h1 = results[:,:,  time_settled:, 0, 1]
    d0 = results[:,:,  time_settled:, 1, 0]
    d1 = results[:,:,  time_settled:, 1, 1]

    variables = {'h1': h1, 'h0': h0, 'd0': d0, 'd1': d1}

    # setup empty arrays for the results
    for i in variables:
        globals()[i + "_LI_mean"] = np.zeros((len(parameter1), len(parameter2)))
        globals()["period_synced_" + i] = np.zeros((len(parameter1), len(parameter2)))
        globals()["period_LI_" + i] = np.zeros((len(parameter1), len(parameter2)))
        globals()["amplitude_synced_" + i] = np.zeros((len(parameter1), len(parameter2)))
        globals()["amplitude_LI_" + i] = np.zeros((len(parameter1), len(parameter2)))

    h_diff_LI_mean = np.zeros((len(parameter1), len(parameter2)))
    d_diff_LI_mean = np.zeros((len(parameter1), len(parameter2)))

    first_zero_index = np.zeros((len(parameter1), len(parameter2)), dtype=int)
    last_zero_index = np.zeros((len(parameter1), len(parameter2)), dtype=int)
    
    # calculate the difference between the two cells 
    h_diff = np.abs(h0 - h1)
    d_diff = np.abs(d0 - d1)

    # split up the results into synchronised and laterally inhibited (LI) parts
    for i in range(len(parameter1)):
        for j in range(len(parameter2)):
            if np.any(d_diff[i,j] < 0.5):

            # Find the first and last index where d_diff is very small 
                zero_indices = np.where(d_diff[i,j] < 0.5)[0]
                first_zero_index[i,j] = zero_indices[0]
                last_zero_index[i,j] = zero_indices[-1] + 1

                # extract the synchronised part of the results and calculate the period and amplitude
                for var in variables:
                    globals()[var + "synced"] = variables[var][i,j, first_zero_index[i,j]:last_zero_index[i,j]]
                    globals()["period_synced_" + var][i,j] = np.nan_to_num(estimate_period_from_peaks(-globals()[var + "synced"], time = np.arange(len(globals()[var + "synced"]))*dt)[0], nan=0.0)
                    globals()["amplitude_synced_" + var][i,j] = np.nan_to_num(estimate_amplitude_from_peaks(globals()[var + "synced"])[0], nan=0.0)

            # if the cell goes into LI, extract the LI part of the results and calculate the period, amplitude and mean value
            elif last_zero_index[i,j] < results.shape[2]:
                tsteps_LI = results.shape[2] - last_zero_index[i,j]
                for var in variables:
                    globals()[var + "LI"] = variables[var][i,j, last_zero_index[i,j]:]
                    globals()["period_LI_" + var][i,j] = np.nan_to_num(estimate_period_from_peaks(-globals()[var + "LI"], time = np.arange(tsteps_LI)*dt)[0], nan=0.0)
                    globals()["amplitude_LI_" + var][i,j] = np.nan_to_num(estimate_amplitude_from_peaks(globals()[var + "LI"])[0], nan=0.0)
                    globals()[var + "_LI_mean"][i,j] = np.mean(globals()[var + "LI"])

                h_diff_LI_mean[i,j] = np.mean(h_diff[i,j, last_zero_index[i,j]:])
                d_diff_LI_mean[i,j] = np.mean(d_diff[i,j, last_zero_index[i,j]:])

    # save calculated values in a list
    result_diff = [h_diff, d_diff]
    result_diff_LI_mean = [h_diff_LI_mean, d_diff_LI_mean]
    result_LI_mean = [h0_LI_mean, h1_LI_mean, d0_LI_mean, d1_LI_mean]
    result_synced_index = [first_zero_index, last_zero_index]
    result_period_synced = [period_synced_h0, period_synced_h1, period_synced_d0, period_synced_d1]
    result_period_LI = [period_LI_h0, period_LI_h1, period_LI_d0, period_LI_d1]
    result_amplitude_synced = [amplitude_synced_h0, amplitude_synced_h1, amplitude_synced_d0, amplitude_synced_d1]
    result_amplitude_LI = [amplitude_LI_h0, amplitude_LI_h1, amplitude_LI_d0, amplitude_LI_d1]

    return result_diff, result_diff_LI_mean, result_LI_mean, result_synced_index, result_period_synced, result_amplitude_synced, result_period_LI, result_amplitude_LI

def analysis_2cells_checkerboard(results, time_settled, parameter1, parameter2):
    """Analyse the results of a two-cell simulations (shape parameter1 x variable 2 x num_tsteps x num_cells x 2) to extract observables such as period, amplitude, and mean values for a seperation 
    of the cell into synchronised and laterally inhibited (LI) parts. This was used for two-cell simulations with a checkerboard initialisation.
    
    Inputs:
    - results: 5D numpy array of shape (parameter1, parameter2, num_tsteps, num_cells, 2) containing the simulation results
    - time_settled: Number of time steps to ignore at the beginning of the simulation
    - parameter1: 1D numpy array of the first parameter values
    - parameter2: 1D numpy array of the second parameter values
    
    Returns:
    - result_diff: List containing the absolute differences between the two cells for h and d for each time step and parameter combination
    - result_mean_diff: List containing the mean of the absolute differences between the two cells for h and d for each parameter combination
    - result_period: List containing the period of h0, h1, d0, and d1 for each parameter combination
    - result_amplitude: List containing the amplitude of h0, h1, d0, and d1 for each parameter combination
    - result_shift: List containing the phase shift between h0 and d0 for each parameter combination"""

    dt = 0.2

    # seperate the results into the different variables
    h0 = results[:,:,  time_settled:, 0, 0]
    h1 = results[:,:,  time_settled:, 0, 1]
    d0 = results[:,:,  time_settled:, 1, 0]
    d1 = results[:,:,  time_settled:, 1, 1]

    variables = {'h1': h1, 'h0': h0, 'd0': d0, 'd1': d1}

    # setup empty arrays for the results
    for i in variables:
        globals()["period_" + i] = np.zeros((len(parameter1), len(parameter2)))
        globals()["amplitude_" + i] = np.zeros((len(parameter1), len(parameter2)))

    # calculate the difference between the two cells 
    h_diff = np.abs(h0 - h1)
    d_diff = np.abs(d0 - d1)

    # calculate the period and amplitude for each variable
    for i in range(len(parameter1)):
        for j in range(len(parameter2)):
            for var in variables:
                # Use the globals() function to access the variable names dynamically   
                globals()["period_" + var][i,j] = np.nan_to_num(estimate_period_from_peaks(-variables[var][i,j], time = np.arange(len(variables[var][i,j]))*dt)[0], nan=0.0)
                globals()["amplitude_" + var][i,j] = np.nan_to_num(estimate_amplitude_from_peaks(variables[var][i,j])[0], nan=0.0)
    
    # save calculated values in a list
    result_diff = [h_diff, d_diff]
    result_mean_diff = mean_diff(result_diff)
    result_period = [period_h0, period_h1, period_d0, period_d1]
    result_amplitude = [amplitude_h0, amplitude_h1, amplitude_d0, amplitude_d1]

    return result_diff, result_mean_diff, result_period, result_amplitude

def analysis_2cells_delay(results, time_settled, parameter1, parameter2, dt=0.2):
    """Analyse the results of a two-cell simulations (shape parameter1 x variable 2 x num_tsteps x num_cells x 2) to extract observables such as period, amplitude, and mean values for a seperation 
    of the cell into synchronised and laterally inhibited (LI) parts. This was used for two-cell simulations with a uniform initialisation and a coupling delay.
    
    Inputs:
    - results: 5D numpy array of shape (parameter1, parameter2, num_tsteps, num_cells, 2) containing the simulation results
    - time_settled: Number of time steps to ignore at the beginning of the simulation to allow the system to settle
    - parameter1: 1D numpy array of the first parameter values
    - parameter2: 1D numpy array of the second parameter values
    - dt: Time step size in seconds (default is 0.2 seconds)
    
    Returns:
    - result_diff: List containing the absolute differences between the two cells for h and d for each time step and parameter combination
    - result_synced_index: List containing the first and last index of the synchronised part of the results for each parameter combination
    - result_period_synced: List containing the period of the synchronised part of h0, h1, d0, and d1 for each parameter combination
    - result_amplitude_synced: List containing the amplitude of the synchronised part of h0, h1, d0, and d1 for each parameter combination
    - result_shift: List containing the phase shift between h0 and d0 for each parameter combination"""

    dt = 0.2

    # seperate the results into the different variables
    h0 = results[:,:,  time_settled:, 0, 0]
    h1 = results[:,:,  time_settled:, 0, 1]
    d0 = results[:,:,  time_settled:, 1, 0]
    d1 = results[:,:,  time_settled:, 1, 1]

    variables = {'h1': h1, 'h0': h0, 'd0': d0, 'd1': d1}

    # setup empty arrays for the results
    for i in variables:
        globals()["period_synced_" + i] = np.zeros((len(parameter1), len(parameter2)))
        globals()["amplitude_synced_" + i] = np.zeros((len(parameter1), len(parameter2)))

    first_zero_index = np.zeros((len(parameter1), len(parameter2)), dtype=int)
    last_zero_index = np.zeros((len(parameter1), len(parameter2)), dtype=int)

    shift_peaks = np.zeros((len(parameter1), len(parameter2)))
    shift_troughs = np.zeros((len(parameter1), len(parameter2)))

    # calculate the difference between the two cells 
    h_diff = np.abs(h0 - h1)
    d_diff = np.abs(d0 - d1)

    for i in range(len(parameter1)):
        for j in range(len(parameter2)):
            # take the part of the results where the difference between the two cells is small, thus the cells are synchronised
            if np.any(d_diff[i,j] < 0.5):
            # Find the first index where d_diff is very small 
                zero_indices = np.where(d_diff[i,j] < 0.5)[0]
                first_zero_index[i,j] = zero_indices[0]
                last_zero_index[i,j] = zero_indices[-1] + 1

                for var in variables:
                    globals()[var + "_synced"] = variables[var][i,j, first_zero_index[i,j]:last_zero_index[i,j]]
                    _, globals()[var + "_peaks"] = estimate_period_from_peaks(globals()[var + "_synced"], time = np.arange(len(globals()[var + "_synced"]))*dt)
                    globals()["period_synced_" + var][i,j], globals()[var + "_troughs"] = estimate_period_from_peaks(-globals()[var + "_synced"], time = np.arange(len(globals()[var + "_synced"]))*dt)
                    globals()["amplitude_synced_" + var][i,j] = estimate_amplitude_from_peaks(globals()[var + "_synced"])[0]

                shift_peaks[i,j] = find_shift(h0_peaks, d0_peaks)
                shift_troughs[i,j] = find_shift(h0_troughs, d0_troughs)
    
    # save calculated values in a list
    result_diff = [h_diff, d_diff]
    result_synced_index = [first_zero_index, last_zero_index]
    result_period_synced = [period_synced_h0, period_synced_h1, period_synced_d0, period_synced_d1]
    result_amplitude_synced = [amplitude_synced_h0, amplitude_synced_h1, amplitude_synced_d0, amplitude_synced_d1]
    result_shift = [shift_peaks, shift_troughs]

    return result_diff, result_synced_index, result_period_synced, result_amplitude_synced, result_shift