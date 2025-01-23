#------------------------------------------------------------------------------
# Pascal Schulthess
# Hubrecht Institute
#
# Functions for pyBoat analysis of time course data
#
# Date: 13-06-2023
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
## Prerequisites
#
#------------------------------------------------------------------------------

using DataFrames, DataFramesMeta
using PyCall

import Pandas

@pyinclude("pyBOAT_0.9.8/pyboat/core.py")

#------------------------------------------------------------------------------
#
## Main function
#
#------------------------------------------------------------------------------

# Inputs
# t: time in minutes
# signal: signal
# sel: signal to analyse (raw, detrended, normed)
# Ts: sampling time
# Tc: cut-off Frequency_1_huency for sinc smoothing (~1.5x expected period) 
# ws: window size of amplitude envelope (~3x expected period)
# ϕ: period range
# power_threshold: wavelet power threshold for ridge detection
# ws_smooth: smoothing window size for ridge

# Outputs
# signals_df, spectrum_df, fourier_df, ridge_df


function pyBoating(
    t::Vector{Float64}, 
    signal::Vector{Float64},
    sel::String,
    Ts::Float64,
    Tc::Float64,
    ws::Float64,
    ϕ::Vector{Float64},
    power_threshold::Float64,
    ws_smooth::Int,
    plot::Bool,
    power_max::Union{Missing,Float64}=missing
    )

    # Determine signal to analyse
    if sel == "raw"
        sig = signal
        signals_df = DataFrame(time=t, signal=sig)
    elseif  sel == "detrended" || sel == "normed"
        # Calculate the trend and detrend the signal
        trend = py"sinc_smooth"(signal, T_c=Tc, dt=Ts)
        detrended = signal - trend

        if sel == "normed"
            # Calculate amplitude envelope of the detrended signal
            # Only explicitly calculate to be store in df
            envelope = py"sliding_window_amplitude"(detrended, window_size=ws, dt=Ts)

            # Normalise detrended signal with envelope
            sig = py"normalize_with_envelope"(detrended, window_size=ws, dt=Ts)

            # Save to df
            signals_df = DataFrame(
                time=t, 
                signal=signal,
                trend=trend,
                detrended=detrended,
                envelope=envelope,
                normed=sig)
        else
            sig=detrended

            # Save to df
            signals_df = DataFrame(
                time=t,
                signal=signal,
                trend=trend,
                detrended=detrended)
        end
    end

    # Wavelet transform
    mod, trans = py"compute_spectrum"(sig, dt=Ts, periods=ϕ)

    # Collect into dataframe
    spectrum_df = DataFrame(
        time=repeat(t, inner=length(ϕ)),
        periods=repeat(periods, outer=length(t)),
        power=reshape(mod, :, 1)[:]
    )

    # Fourier transform
    fourier_df = _fourier(sig, Ts)

    # Ridge detection
    ridge_df = _ridge(t, sig, mod, trans, ϕ, power_threshold, ws_smooth)

    # Plot
    if plot
        fig = _plt(sel, t, sig, ϕ, mod, ridge_df, power_max)
        return signals_df, spectrum_df, fourier_df, ridge_df, fig
    else
        return signals_df, spectrum_df, fourier_df, ridge_df
    end
end

# Compute fourier transform
function _fourier(signal::Vector{Float64}, Ts::Float64)
    freqs, power = py"compute_fourier"(signal, dt=Ts)
    fourier_df = DataFrame(
        frequency=freqs,
        power=power
    )
    return fourier_df
end

# Ridge detection
function _ridge(
    t::Vector{Float64},
    signal::Vector{Float64},
    mod::Matrix{Float64},
    trans::Matrix{ComplexF64},
    ϕ::Vector{Float64},
    power_threshold::Float64,
    ws_smooth::Int)

    # Ridge detection
    r = py"get_maxRidge_ys"(mod)
    r_eval = py"eval_ridge"(
        ridge_y=r,
        transform=trans,
        signal=signal,
        periods=ϕ,
        tvec=t,
        power_thresh=power_threshold,
        smoothing_wsize=ws_smooth
        )
    ridge_df = r_eval |>
        Pandas.DataFrame |>
        DataFrame

    if isempty(ridge_df)
        push!(allowmissing!(ridge_df), repeat([missing], size(ridge_df, 2)))
    end
    return ridge_df
end

# Plot signal, spectrum, and ridge
function _plt(
    sel::String,
    t::Vector{Float64},
    signal::Vector{Float64},
    ϕ::Vector{Float64},
    mod::Matrix{Float64},
    r::DataFrame,
    power_max::Union{Missing,Float64}=missing)

    fig=Figure(
        resolution=(600,700)
        )
    ax_signal = Axis(
        fig[1, 1],
        ylabel=sel,
        limits=(minimum(t), maximum(t), nothing, nothing)
    )

    ax_spectrum = Axis(
        fig[2, 1],
        xlabel="time",
        ylabel="period",
        limits=(minimum(t), maximum(t), nothing, nothing)
    )

    # Plot signal
    scatterlines!(ax_signal, t, signal)

    # Plot spectrum
    pwr_max = ismissing(power_max) ? maximum(mod) : power_max
    hm = heatmap!(
        ax_spectrum, t, ϕ, mod', colorrange=(0, pwr_max)
    )
    
    # Plot ridge
    scatter!(ax_spectrum, r.time, r.periods, 
        color=:tomato, markersize=4)
    
    # Colormap
    Colorbar(
        fig[2, 2], hm,
        label="Wavelet power",
        flipaxis=true
        )

    rowsize!(fig.layout, 2, Relative(2 / 3))
    colsize!(fig.layout, 1, Relative(1))

    return fig
end