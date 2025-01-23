#------------------------------------------------------------------------------
# Pascal Schulthess
# Hubrecht Institute
# modified by Ina Sonnen
#
# Data analysis of Hes1 single cell tracks in spread-outs
#
# Date: 16-11-2023
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
## Prerequisites
#
#------------------------------------------------------------------------------

using CairoMakie
using ShiftedArrays
using KernelDensity
using CSV
using DataFrames, DataFramesMeta
using Statistics, StatsBase
using RollingFunctions
using GLM
using HypothesisTests
using Distributions
using Peaks

# Jump to code folder
if !occursin("PS23_Code", pwd())
    cd("PS23_Code")
end

# Include my functions
include("PS23_pyBoat_analysis.jl")

# Define figure and table path
fig_path = "../PS23_Figures/PS23_Single-cell-tracks/PS23_Single-cell-tracks_"
tab_path = "../PS23_Tables/PS23_Single-cell-tracks/PS23_Single-cell-tracks_"

# Plotting theme
mytheme = Attributes(
    Axis=(
        xminorticksvisible=true,
        xminorgridvisible=true,
        yminorticksvisible=true,
        yminorgridvisible=true,
        rightspinevisible=false,
        topspinevisible=false
    )
)
set_theme!(mytheme)

# Annoyingly ShiftedArrays requires this
function lag(x)
    return ShiftedArrays.lag(x)
end
function lead(x)
    return ShiftedArrays.lead(x)
end

# Jitter points
jitter(n::Real, factor=0.1) = n + (0.5 - rand()) * factor

# My color palettes and linestyles
mycol_default = Makie.wong_colors();
mycol_viridis(n::Union{Int64,UnitRange}, len::Int64) = cgrad(:viridis, len, categorical=true)[n]

# p-value to asterisk
function _pvalue_to_string(p)

    if p > 0.05
        out = "n.s."
    elseif p <= 0.05 && p > 0.01
        out = "⋆"
    elseif p <= 0.01 && p > 0.001
        out = "⋆⋆"
    elseif p <= 0.001 && p > 0.0001
        out = "⋆⋆⋆"
    elseif p <= 0.0001
        out = "⋆⋆⋆⋆"
    end
end

# Function to determine the full width at half-maxium as a measure of noisyness
# of a distribution of observations
function _fwhm(data)
    # Estimate kernel density and maximum
    kde_data = kde(data)
    halfmax_density = maximum(kde_data.density) / 2

    # Put estimates in dataframe
    df = DataFrame(x=kde_data.x, den=kde_data.density)
    df_halfmax = @rsubset(df, :den >= halfmax_density)

    # Return with
    return df_halfmax.x[end] - df_halfmax.x[1]
end

#------------------------------------------------------------------------------
#
## Load data from csv
#
#------------------------------------------------------------------------------

# Define column names
col_names = [
    "Region",
    "Sp_ID",
    "Sp_N_Links",
    "Sp_Int_Cent_Hes1",
    "Sp_Int_Cent_H2b",
    "Sp_Frame",
    "Sp_Int_Mean_Hes1",
    "Sp_Int_Std_Hes1",
    "Sp_Int_Min_Hes1",
    "Sp_Int_Max_Hes1",
    "Sp_Int_Med_Hes1",
    "Sp_Int_Sum_Hes1",
    "Sp_Int_Mean_H2b",
    "Sp_Int_Std_H2b",
    "Sp_Int_Min_H2b",
    "Sp_Int_Max_H2b",
    "Sp_Int_Med_H2b",
    "Sp_Int_Sum_H2b",
    "Sp_Pos_X_um",
    "Sp_Pos_Y_um",
    "Sp_Pos_Z_um",
    "__drop_1",
    "__drop_2",
    "Sp_Rad_um",
    "Sp_Tr_ID",
    "Tr_N_Sp"]

# Path to the csvs
data_dir = "../PS23_Data/Spreadouts_singlecell_tracking/"

# List Hes1/Hes7 folders
lst_files = readdir(data_dir)
lst_files = lst_files[lst_files.!==".DS_Store"]

# Seperate out representative tracks
rep_files = lst_files[occursin.("figure", lst_files).==1]
lst_files = lst_files[occursin.("figure", lst_files).==0]

# Initialise
data_all = DataFrame()

# Counter
n = 1

# Loop over files
for i in eachindex(lst_files)

    println("File: " * string(i) * "/" * string(length(lst_files)))

    # Get descriptors from file names
    date_i, medium_id_i, _ = split(lst_files[i], "_")

    # File path
    data_path_i = data_dir * lst_files[i]

    # Load data
    data_raw = CSV.read(data_path_i, DataFrame, header=col_names, drop=[22, 23], skipto=4)

    # Add time and descriptors
   data_i = @chain data_raw begin
        @rtransform :Time_min = :Sp_Frame * 10.0
        @rtransform :Experiment = date_i * "_" * medium_id_i
        @rtransform :Date = date_i
        @rtransform :Medium = uppercase(medium_id_i[1:end-3])
        @rtransform :Position = medium_id_i[end-2:end]
        # Unique Spot_ID and Spot_Track_ID
        @rtransform :Sp_ID = string(n) * "_" * string(:Sp_ID)
        @rtransform :Sp_Tr_ID = string(n) * "_" * string(:Sp_Tr_ID)
        # Differentiation flag
        @rtransform :Differentiated = :Region == "SM" ? "yes" : "no"
        @rtransform :Region = :Region == "SM" ? "PSM" : :Region
    end

    # Combine
    global data_all = [data_all; data_i]
    global n += 1

end

# Initialise
data_rep = DataFrame()

# Counter
n = 1

# Loop over files
for i in eachindex(rep_files)

    println("File: " * string(i) * "/" * string(length(rep_files)))

    # Get descriptors from file names
    date_i, medium_id_i, _ = split(rep_files[i], "_")

    # File path
    data_path_i = data_dir * rep_files[i]

    # Load data
    data_raw = CSV.read(data_path_i, DataFrame, header=col_names, drop=[22, 23], skipto=4)

    # Add time and descriptors
    data_i = @chain data_raw begin
        @rtransform :Time_min = :Sp_Frame * 10.0
        @rtransform :Experiment = date_i * "_" * medium_id_i
        @rtransform :Date = date_i
        @rtransform :Medium = uppercase(medium_id_i[1:end-3])
        @rtransform :Position = medium_id_i[end-2:end]
        # Unique Spot_ID and Spot_Track_ID
        @rtransform :Sp_ID = string(n) * "_" * string(:Sp_ID)
        @rtransform :Sp_Tr_ID = string(n) * "_" * string(:Sp_Tr_ID)
        # Differentiation flag
        @rtransform :Differentiated = :Region == "SM" ? "yes" : "no"
        @rtransform :Region = :Region == "SM" ? "PSM" : :Region
    end

    # Combine
    global data_rep = [data_rep; data_i]
    global n += 1

end

## Post-process data ----------------------------------------------------------
function _data_postprocess(data; ra_window = 5)

    data_out = @chain data begin
        # Ordering
        sort(_, [order(:Medium, rev=true), order(:Region, rev=true), :Sp_Tr_ID, :Time_min])
        # Assign zplanes: -5-5 -> 1, 5-15 -> 2, 15-25 -> 3, ...
        @rtransform :Sp_Pos_Z_Pl = Int(round(:Sp_Pos_Z_um / 10)) + 1
        # Normalise Hes1 by the mean
        groupby(_, :Sp_Tr_ID)
        @transform :Sp_Int_Cent_Hes1_Norm_Mean = :Sp_Int_Cent_Hes1 ./ mean(:Sp_Int_Cent_Hes1)
        # Rolling average
        groupby(_, :Sp_Tr_ID)
        @transform :Sp_Int_Cent_Hes1_RA = [repeat([missing], div(ra_window, 2)); rollmean(:Sp_Int_Cent_Hes1, ra_window); repeat([missing], div(ra_window, 2))]
        groupby(_, :Sp_Tr_ID)
        @transform :Sp_Int_Cent_Hes1_Norm_Mean_RA = [repeat([missing], div(ra_window, 2)); rollmean(:Sp_Int_Cent_Hes1_Norm_Mean, ra_window); repeat([missing], div(ra_window, 2))]
        # Add time relative to the differentiation time point
        @rtransform :Diff_Time_min = :Differentiated == "yes" ? :Time_min : missing
        groupby(_, [:Experiment, :Sp_Tr_ID])
        @transform :Diff_Time_min = sum(skipmissing(:Diff_Time_min), init=0) > 0 ? minimum(skipmissing(:Diff_Time_min)) : maximum(:Time_min)
        @rtransform :rel_Time_min = :Time_min - :Diff_Time_min
    end
    return data_out
end

data_all_pp = _data_postprocess(data_all)
data_rep_pp = _data_postprocess(data_rep)

#------------------------------------------------------------------------------
#
## Plotting
#
#------------------------------------------------------------------------------

## Plot individual tracks - each track one facet ------------------------------
# with differentiation points highlihgted
function _plot_ind_tracks_facetted(obs::String)

    # Select plot data
    if obs == "Hes1"
        y_name = "Sp_Int_Cent_Hes1"
        y_axis_name = "Hes1"
    elseif obs == "Hes1_RA"
        y_name = "Sp_Int_Cent_Hes1_RA"
        y_axis_name = "Hes1"
    elseif obs == "Hes1mean"
        y_name = "Sp_Int_Cent_Hes1_Norm_Mean"
        y_axis_name = "Norm. Hes1"
    elseif obs == "Hes1mean_RA"
        y_name = "Sp_Int_Cent_Hes1_Norm_Mean_RA"
        y_axis_name = "Norm. Hes1"
    end

    fig = Figure(
        resolution=(2000, 1200),
        font="Helvetica Neue"
    )

    # Unique tracks
    unique_tracks = unique(data_all_pp.Sp_Tr_ID)
    n_trk = length(unique_tracks)
    unique_zplanes = sort(unique(data_all_pp.Sp_Pos_Z_Pl))

    # Define number of columns and rows
    n_col = 13
    n_row = n_trk ÷ n_col

    # Axes
    axs = [Axis(
        fig[fldmod1(i, n_col)...],
        xlabel="Time (h)",
        ylabel=y_axis_name * " (a.u.)",
        xticks=0:6:24,
        xminorticks=0:1:24,
        limits=(-1, 25, nothing, nothing)
    ) for i in eachindex(unique_tracks)]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        # Extract data
        data_i = @chain data_all_pp begin
            @rsubset :Sp_Tr_ID == unique_tracks[i]
            @rtransform :Time_h = :Time_min ./ 60.0
            @orderby :Time_h
        end

        # Axis
        ax.title = data_i.Region[1]
        ax.subtitle = data_i.Date[1] * " " * data_i.Medium[1] * " " * data_i.Position[1]

        if i ∉ 1:n_col:n_trk
            ax.ylabelvisible = false
        end
        if i ∉ (n_trk-n_col+1):n_trk
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        # Plot
        lines!(
            ax, data_i.Time_h, data_i[!, y_name],
            color=:black, linewidth=0.7)

        # Loop over zplanes
        for j in eachindex(unique_zplanes)

            # Extract data
            data_j_nodiff = @rsubset(data_i, :Sp_Pos_Z_Pl == unique_zplanes[j] && :Differentiated == "no")
            data_j_yesdiff = @rsubset(data_i, :Sp_Pos_Z_Pl == unique_zplanes[j] && :Differentiated == "yes")

            if !isempty(data_j_nodiff)
                # Plot
                scatter!(
                    ax, data_j_nodiff.Time_h, data_j_nodiff[!, y_name],
                    color=mycol_viridis(j, length(unique_zplanes)), markersize=5, label=string(unique_zplanes[j]))

            end
            if !isempty(data_j_yesdiff)
                # Plot
                scatter!(
                    ax, data_j_yesdiff.Time_h, data_j_yesdiff[!, y_name],
                    color=mycol_viridis(j, length(unique_zplanes)), markersize=5,
                    strokewidth=1, strokecolor=mycol_default[6])
            end
        end
    end

    mylegend = [PolyElement(color=mycol_viridis(i, length(unique_zplanes))) for i in eachindex(unique_zplanes)]
    fig[:, end+1] = Legend(fig, [mylegend], [string.(unique_zplanes)], ["z-plane"], framevisible=false)

    if (n_trk % n_col) == 0
        [rowgap!(fig.layout, i, Relative(0.02)) for i in 1:(n_row)]
    else
        [rowgap!(fig.layout, i, Relative(0.02)) for i in 1:(n_row-1)]
        rowgap!(fig.layout, n_row, Relative(-0.05))
    end

    fig

end

# Save
fig_name = "Hes1-absolute_tracks-facetted_color-zplane-diffpoint"
save(fig_path * fig_name * ".pdf", _plot_ind_tracks_facetted("Hes1"))

fig_name = "Hes1-absolute-rollavg_tracks-facetted_color-zplane-diffpoint"
save(fig_path * fig_name * ".pdf", _plot_ind_tracks_facetted("Hes1_RA"))

fig_name = "Hes1-mean-norm_tracks-facetted_color-zplane-diffpoint"
save(fig_path * fig_name * ".pdf", _plot_ind_tracks_facetted("Hes1mean"))

fig_name = "Hes1-mean-norm-rollavg_tracks-facetted_color-zplane"
save(fig_path * fig_name * ".pdf", _plot_ind_tracks_facetted("Hes1mean_RA"))


## Plot individual tracks - each track one facet ------------------------------
# with differentiation points colored differently but only for NB medium
function _plot_ind_tracks_facetted_nb_only(obs::String)

    if obs == "abs"
        res=(1800, 1000)
        y_obs = "Sp_Int_Cent_Hes1_RA"
        y_label = "Hes1 (a.u.)"
        y_min = nothing
        y_max = nothing
    elseif obs == "norm"
        res=(1200, 800)
        y_obs = "Sp_Int_Cent_Hes1_Norm_Mean_RA"
        y_label = "Norm.\nHes1 (a.u.)"
        y_min = 0.0
        y_max = 2.8
    end

    fig = Figure(
        resolution=res,
        font="Helvetica Neue"
    )

    # Extract data
    data_plt = @chain data_all_pp begin
        @rsubset :Medium == "NB"
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
    end

    # Unique tracks
    unique_tracks = unique(data_plt.Tr_ID)
    n_trk = length(unique_tracks)

    # Define number of columns and rows
    n_col = 10
    n_row = n_trk ÷ n_col

    # Axes
    axs = [Axis(
        fig[fldmod1(i, n_col)...],
        xlabel="Time (h)",
        ylabel=y_label,
        xticks=0:6:24,
        xminorticks=0:1:24,
        limits=(-1, 25, y_min, y_max)
    ) for i in eachindex(unique_tracks)]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        # Extract data
        data_i = @chain data_plt begin
            @rsubset :Tr_ID == unique_tracks[i]
            @rtransform :Time_h = :Time_min ./ 60.0
            @orderby :Time_h
        end

        data_i_nodiff = @rsubset(data_i, :Differentiated == "no")
        data_i_yesdiff = @rsubset(data_i, :Differentiated == "yes")

        # Axis
        ax.title = data_i.Region[1] * " (#" * string(data_i.Tr_ID[1]) * ")"

        if i ∉ 1:n_col:n_trk
            ax.ylabelvisible = false
            if obs == "norm"
                ax.yticklabelsvisible = false
            end

        end
        if i ∉ (n_trk-n_col+1):n_trk
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        if data_i.Region[1] == "PSM"
            my_cols_i = mycol_default[[1,5]]
        elseif data_i.Region[1] == "NT"
            my_cols_i = mycol_default[[2,7]]
        end

        # Plot
        lines!(
            ax, data_i_nodiff.Time_h, data_i_nodiff[!, y_obs],
            color=my_cols_i[1])
        lines!(
            ax, data_i_yesdiff.Time_h, data_i_yesdiff[!, y_obs],
            color=my_cols_i[2])

        # Loop over zplanes
    end

    if (n_trk % n_col) == 0
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row)]
    else
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row-1)]
        if obs == "abs"
            rowgap!(fig.layout, n_row, Relative(-0.05))
        elseif obs == "norm"
            rowgap!(fig.layout, n_row, Relative(-0.075))
        end
    end
    colgap!(fig.layout, Relative(0.005))

    fig

end

# Save
fig_name = "Hes1-absolute-rollavg_tracks-facetted_color-region_NB-only"
save(fig_path * fig_name * ".pdf", _plot_ind_tracks_facetted_nb_only("abs"))

fig_name = "Hes1-mean-norm-rollavg_tracks-facetted_color-region_NB-only"
save(fig_path * fig_name * ".pdf", _plot_ind_tracks_facetted_nb_only("norm"))


## Plot Hes1 absolute per region in NB as heatmap -----------------------------
function _plot_heatmap(signal::String, region::String; time_sel::String, sort::String)

    # Prepare data
    data = @chain data_all_pp begin
        @rsubset :Region == region && :Medium == "NB"
        groupby(_, :Sp_Tr_ID)
        @transform :min_Time = time_sel == "absolute" ? minimum(:Time_min) : minimum(:rel_Time_min)
        @rtransform :Time = time_sel == "absolute" ? :Time_min : :rel_Time_min
    end

    fig = Figure(
        resolution=(300, 400),
        font="Helvetica Neue"
    )

    ax = Axis(
        fig[1, 1],
        xlabel=time_sel == "absolute" ? "Time (h)" : "Rel. time (h)",
        ylabel="Tracks",
        xticks=-24:6:24,
        xgridvisible=false,
        ygridvisible=false,
        xminorgridvisible=false,
        yminorgridvisible=false,
        xminorticksvisible=false,
        yminorticksvisible=false,
        yticklabelsvisible=false,
        yticksvisible=false,
        leftspinevisible=false,
        # limits=(-1, 25, nothing, nothing)
    )

    if sort == "t0"
        data_plt = @chain data begin
            @orderby(_, :min_Time, :Time)
            groupby(_, :Sp_Tr_ID)
            @transform :Tr_ID = $groupindices
        end
    elseif sort == "t0diff"
        data_plt = @chain data begin
            @orderby(_, :min_Time, :Diff_Time_min, :Time)
            groupby(_, :Sp_Tr_ID)
            @transform :Tr_ID = $groupindices
        end
    else
        data_plt = @chain data begin
            groupby(_, :Sp_Tr_ID)
            @transform :Tr_ID = $groupindices
        end
    end
    
    if region == "PSM"
        hm = heatmap!(
            ax, data_plt.Time ./ 60.0, data_plt.Tr_ID, data_plt[:, signal],
            colorrange=(0, maximum(data_plt[:, signal])))

        # define colorbar ticks
        if signal == "Sp_Int_Cent_Hes1"
            cb_ticks = 0:100:maximum(data_plt[:, signal])
        elseif signal == "Sp_Int_Cent_Hes1_Norm_Mean"
            cb_ticks = 0:1:maximum(data_plt[:, signal])
        end

        # Differentiation times
        if time_sel == "absolute"
            data_diff = @rsubset(data_plt, :Time == :Diff_Time_min)
            scatter!(
                ax, data_diff.Diff_Time_min ./ 60.0, data_diff.Tr_ID, 
                color=:white, markersize=7
                )
        else
            scatter!(
                ax, repeat([0.0], length(unique(data_plt.Tr_ID))), unique(data_plt.Tr_ID),
                color=:white, markersize=7
                )
        end
    elseif region == "NT"
        hm = heatmap!(
            ax, data_plt.Time_min ./ 60.0, data_plt.Tr_ID, data_plt[:, signal],
            colorrange=(0, maximum(data_plt[:, signal])))

        # define colorbar ticks
        if signal == "Sp_Int_Cent_Hes1"
            cb_ticks = 0:200:maximum(data_plt[:, signal])
        elseif signal == "Sp_Int_Cent_Hes1_Norm_Mean"
            cb_ticks = 0:1:maximum(data_plt[:, signal])
        end
        
    end

    if signal == "Sp_Int_Cent_Hes1"
        colbar_label = "Hes1 (a.u.)"
    elseif signal == "Sp_Int_Cent_Hes1_Norm_Mean"
        colbar_label = "Norm. Hes1 (a.u.)"
    end

    Colorbar(
        fig[:, end+1], hm, label=colbar_label, ticks=cb_ticks,
        height=100, width=10, tellheight=false
        )
    colgap!(fig.layout, Relative(0.02))

    fig
end

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_PSM_abs-time_unsort"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "PSM", time_sel="absolute", sort="none"))

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_PSM_abs-time_sort-t0"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "PSM", time_sel="absolute", sort="t0"))

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_PSM_abs-time_sort-t0-diff"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "PSM", time_sel="absolute", sort="t0diff"))

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_PSM_rel-time_unsort"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "PSM", time_sel="relative", sort="none"))

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_PSM_rel-time_sort-t0"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "PSM", time_sel="relative", sort="t0"))

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_NT_abs-time_unsort"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "NT", time_sel="absolute", sort="none"))

fig_name = "Hes1-absolute_tracks-heatmap_NB-only_NT_abs-time_sort"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1", "NT", time_sel="absolute", sort="t0"))

fig_name = "Hes1-mean-norm_tracks-heatmap_NB-only_PSM_rel-time_sort-t0"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1_Norm_Mean", "PSM", time_sel="relative", sort="t0"))

fig_name = "Hes1-mean-norm_tracks-heatmap_NB-only_NT_abs-time_sort"
save(fig_path * fig_name * ".pdf", _plot_heatmap("Sp_Int_Cent_Hes1_Norm_Mean", "NT", time_sel="absolute", sort="t0"))

## Plot z position - each track one facet -------------------------------------
# with differentiation points highlihgted
function _plot_zpos_tracks_facetted()

    fig = Figure(
        resolution=(2000, 1200),
        font="Helvetica Neue"
    )

    # Unique tracks
    unique_tracks = unique(data_all_pp.Sp_Tr_ID)
    n_trk = length(unique_tracks)
    unique_zplanes = sort(unique(data_all_pp.Sp_Pos_Z_Pl))

    # Define number of columns and rows
    n_col = 13
    n_row = n_trk ÷ n_col

    # Axes
    axs = [Axis(
        fig[fldmod1(i, n_col)...],
        xlabel="Time (h)",
        ylabel="z (μm)",
        xticks=0:6:24,
        yticks=0:20:60,
        xminorticks=0:1:24,
        limits=(-1, 25, -5, 65),
        yreversed=true
    ) for i in eachindex(unique_tracks)]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        # Extract data
        data_i = @chain data_all_pp begin
            @rsubset :Sp_Tr_ID == unique_tracks[i]
            @rtransform :Time_h = :Time_min ./ 60.0
            @orderby :Time_h
        end

        # Axis
        ax.title = data_i.Region[1]
        ax.subtitle = data_i.Date[1] * " " * data_i.Medium[1] * " " * data_i.Position[1]

        if i ∉ 1:n_col:n_trk
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
        end
        if i ∉ (n_trk-n_col+1):n_trk
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        # Plot
        lines!(
            ax, data_i.Time_h, data_i.Sp_Pos_Z_um,
            color=:black, linewidth=0.7)

        # Loop over zplanes
        for j in eachindex(unique_zplanes)

            # Extract data
            data_j_nodiff = @rsubset(data_i, :Sp_Pos_Z_Pl == unique_zplanes[j] && :Differentiated == "no")
            data_j_yesdiff = @rsubset(data_i, :Sp_Pos_Z_Pl == unique_zplanes[j] && :Differentiated == "yes")

            # Plot
            if !isempty(data_j_nodiff)
                scatter!(
                    ax, data_j_nodiff.Time_h, data_j_nodiff.Sp_Pos_Z_um,
                    color=mycol_viridis(j, length(unique_zplanes)), markersize=5, label=string(unique_zplanes[j]))
            end
            if !isempty(data_j_yesdiff)
                # Plot
                scatter!(
                    ax, data_j_yesdiff.Time_h, data_j_yesdiff.Sp_Pos_Z_um,
                    color=mycol_viridis(j, length(unique_zplanes)), markersize=5,
                    strokewidth=0.5, strokecolor=mycol_default[6])
            end
        end
    end

    mylegend = [PolyElement(color=mycol_viridis(i, length(unique_zplanes))) for i in eachindex(unique_zplanes)]
    fig[:, end+1] = Legend(fig, [mylegend], [string.(unique_zplanes)], ["z-plane"], framevisible=false)

    if (n_trk % n_col) == 0
        [rowgap!(fig.layout, i, Relative(0.02)) for i in 1:(n_row)]
    else
        [rowgap!(fig.layout, i, Relative(0.02)) for i in 1:(n_row-1)]
        rowgap!(fig.layout, n_row, Relative(-0.05))
    end
    colgap!(fig.layout, Relative(0.02))

    fig

end

# Save
fig_name = "Z-position_tracks-facetted_color-zplane-diffpoint"
save(fig_path * fig_name * ".pdf", _plot_zpos_tracks_facetted())


## Plot zplane-normalised z-position per experiment ---------------------------
function _plot_z_pos_norm_zplane_per_exp()

    # Plot data
    data_plt = @chain data_all_pp begin
        @rtransform :Sp_Pos_Norm = :Sp_Pos_Z_um / :Sp_Pos_Z_Pl
    end

    # Unique experiments
    unique_exp = unique(data_plt.Experiment)

    fig = Figure(
        resolution=(500, 300),
        font="Helvetica Neue"
    )

    # Axes
    ax = Axis(
        fig[1, 1],
        xlabel="z position/plane (a.u.)",
        ylabel="Experiment",
        yticks=(collect(eachindex(unique_exp)), unique_exp),
        yminorticksvisible=false,
        yminorgridvisible=false
    )

    # Loop over experiments
    for i in eachindex(unique_exp)

        # Extract data
        data_i = @rsubset(data_plt, :Experiment == unique_exp[i])

        # Plot individual observations
        scatter!(
            ax, data_i.Sp_Pos_Z_um, jitter.(repeat([i], length(data_i.Experiment)), 0.25),
            color=(:black, 0.1), strokewidth=1, strokecolor=:black
        )
    end

    fig

end

# Save
fig_name = "Z-position-norm-zplane_per-experiment"
save(fig_path * fig_name * ".pdf", _plot_z_pos_norm_zplane_per_exp())


# Plot distribution of cells per z-plane per region ---------------------------
function _plot_z_plane_dist_per_region()

    # Calculate distribution by hand
    data_plt = @chain data_all_pp begin
        @rtransform :Region = :Differentiated == "yes" ? "SM" : :Region
        @rtransform :Time_h = :Time_min ./ 60.0
        groupby(_, :Region)
        @transform :Sp_Nbr_Tot = length(:Time_h)
        groupby(_, [:Region, :Sp_Pos_Z_Pl, :Sp_Nbr_Tot])
        @combine :Sp_Nbr_Z_Pl = length(:Time_h)
        @orderby :Region :Sp_Pos_Z_Pl
        @rtransform :Sp_Nbr_Occurance = :Sp_Nbr_Z_Pl / :Sp_Nbr_Tot
    end

    fig = Figure(
        resolution=(450, 250),
        font="Helvetica Neue"
    )

    # Axes
    ax = Axis(
        fig[1, 1],
        xlabel="z-plane",
        ylabel="Proportion of cells",
        xticks=unique(data_plt.Sp_Pos_Z_Pl),
        yticks=0.0:0.1:0.5,
        xminorgridvisible=false,
        xminorticksvisible=false,
        yminorticksvisible=false,
        yminorgridvisible=false
    )

    unique_region = unique(data_plt.Region)[[2, 3, 1]]
    for i in eachindex(unique_region)

        data_i = @rsubset(data_plt, :Region == unique_region[i])

        # Offset
        offset = 0.25
        if unique_region[i] == "PSM"
            offset_i = -1 * offset
        elseif unique_region[i] == "SM"
            offset_i = 0
        elseif unique_region[i] == "NT"
            offset_i = offset
        end

        barplot!(
            ax, data_i.Sp_Pos_Z_Pl .+ offset_i, data_i.Sp_Nbr_Occurance,
            color=(mycol_default[i], 0.5), width=0.2,
            strokewidth=1.5, strokecolor=mycol_default[i], label=unique_region[i])
    end

    fig[:, end+1] = Legend(fig, ax, "Region", framevisible=false)

    fig
end

# Save
fig_name = "Proportion-cells_per-zplane_per-region"
save(fig_path * fig_name * ".pdf", _plot_z_plane_dist_per_region())


# Correlate absolute levels vs z-position - coloured by z-plane --------------------
function _plot_abs_vs_z_pos_col_zplane()

    fig = Figure(
        resolution=(425, 350),
        font="Helvetica Neue"
    )

    # Axes
    ax_hes1 = Axis(
        fig[1, 1],
        xlabel="z position (μm)",
        ylabel="Hes1 (a.u.)",
        xticks=0:10:70,
        xlabelvisible=false,
        xticklabelsvisible=false,
        xminorticksvisible=false
    )

    ax_h2b = Axis(
        fig[2, 1],
        xlabel="z position (μm)",
        ylabel="H2b (a.u.)",
        xticks=0:10:70
    )

    # Unique zplanes
    unique_zplane = sort(unique(data_all_pp.Sp_Pos_Z_Pl))

    # Colopmap for this plot
    colmap = mycol_viridis(1:length(unique_zplane), length(unique_zplane))

    # Loop over zplanes
    for i in unique_zplane

        # Extract data
        data_i = @rsubset(data_all_pp, :Sp_Pos_Z_Pl == i)

        # Plot individual observations
        scatter!(
            ax_hes1, data_i.Sp_Pos_Z_um, data_i.Sp_Int_Cent_Hes1,
            color=(colmap[i], 0.1), strokecolor=colmap[i], strokewidth=1,
            label=string(unique_zplane[i]))
        scatter!(
            ax_h2b, data_i.Sp_Pos_Z_um, data_i.Sp_Int_Cent_H2b,
            color=(colmap[i], 0.1), strokecolor=colmap[i], strokewidth=1,
            label=string(unique_zplane[i]))

        # Plot means per z-plane
        scatter!(
            ax_hes1,
            (i - 1) * 10, mean(data_i.Sp_Int_Cent_Hes1),
            marker='⬦', markersize=30, color=mycol_default[6])
        errorbars!(ax_hes1,
            [i - 1] * 10, [mean(data_i.Sp_Int_Cent_Hes1)], [std(data_i.Sp_Int_Cent_Hes1)],
            whiskerwidth=10, color=mycol_default[6])

        scatter!(
            ax_h2b,
            (i - 1) * 10, mean(data_i.Sp_Int_Cent_H2b),
            marker='⬦', markersize=30, color=mycol_default[6])
        errorbars!(ax_h2b,
            [i - 1] * 10, [mean(data_i.Sp_Int_Cent_H2b)], [std(data_i.Sp_Int_Cent_H2b)],
            whiskerwidth=10, color=mycol_default[6])
    end

    fig[:, end+1] = Legend(fig, ax_h2b, "z-plane", framevisible=false)
    rowgap!(fig.layout, Relative(0.02))

    fig

end

# Save
fig_name = "Hes1-absolute_H2b-absolute_vs_z-position_colored-by-zplane"
save(fig_path * fig_name * ".pdf", _plot_abs_vs_z_pos_col_zplane())


## Plot absolute Hes1 vs H2b levels - facetted by region + regression ---------
function _plot_abs_hes1_vs_h2b_facet_region()

    # Prepare data
    data_plt = @rtransform(data_all_pp, :Region = :Differentiated == "yes" ? "SM" : :Region)

    # Unique 
    unique_region = unique(data_plt.Region)
    unique_medium = unique(data_plt.Medium)

    fig = Figure(
        resolution=(550, 350),
        font="Helvetica Neue"
    )

    # Axes
    axs = [Axis(
        fig[i, j],
        xlabel="H2b (a.u.)",
        ylabel="Hes1 (a.u.)",
        limits=(-100, 3500, -50, 750)
    ) for i in eachindex(unique_medium) for j in eachindex(unique_region)]

    # translate i to k and l for calling region and medium
    kl = [[k, l] for k in eachindex(unique_medium) for l in eachindex(unique_region)]

    # Loop over zplanes
    for (i, ax) in enumerate(axs)

        k = kl[i][1]
        l = kl[i][2]

        # Extract data
        data_i = @rsubset(data_plt, :Medium == unique_medium[k] && :Region == unique_region[l])

        # Axis
        ax.title = unique_medium[k] * " " * unique_region[l]
        if i ∉ [1, 4]
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
        end
        if i ∉ 4:6
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        # Linear regression
        lm_h2b_hes1 = lm(@formula(Sp_Int_Cent_Hes1 ~ Sp_Int_Cent_H2b), data_i)

        # Correlation coefficient
        r_i = round(cor(data_i.Sp_Int_Cent_H2b, data_i.Sp_Int_Cent_Hes1), digits=2)

        # Plot individual observations
        scatter!(
            ax, data_i.Sp_Int_Cent_H2b, data_i.Sp_Int_Cent_Hes1,
            color=(mycol_default[1], 0.1), strokecolor=mycol_default[1], strokewidth=1)

        ablines!(ax, coef(lm_h2b_hes1)[1], coef(lm_h2b_hes1)[2],
            color=mycol_default[2], linewidth=3)

        text!(ax, 2500, 600, text="r=" * string(r_i), align=(:center, :center))
    end

    colgap!(fig.layout, Relative(0.02))
    rowgap!(fig.layout, Relative(0.02))

    fig

end

# Save
fig_name = "H2b-vs-Hes1-absolute_linear-regression_regions-facetted"
save(fig_path * fig_name * ".pdf", _plot_abs_hes1_vs_h2b_facet_region())


## Compare absolute Hes1 before against after differentiation and with NT -----
function _plot_abs_hes1_before_vs_after_diff_w_NT()

    # Collect data
    data_psm = @chain data_all_pp begin
        # Only PSM data in NB
        @rsubset :Region == "PSM" && :Medium == "NB"
        # Only tracks that have a differentiation point
        groupby(_, :Sp_Tr_ID)
        @transform :Diff_present = length(unique(:Differentiated)) == 2 ? "yes" : "no"
    end

    data_nt_tmp = @rsubset(data_all_pp, :Region == "NT" && :Medium == "NB")

    data_nt = @chain data_nt_tmp begin
        groupby(_, [:Region, :Medium, :Sp_Tr_ID])
        @combine @astable begin
            :Sp_Int_Cent_Hes1 = mean(skipmissing(:Sp_Int_Cent_Hes1))
        end
        @rsubset !isnan(:Sp_Int_Cent_Hes1)
    end

    data_before_tmp = @rsubset(data_psm, :Differentiated == "no")

    data_before = @chain data_before_tmp begin
        groupby(_, [:Region, :Medium, :Sp_Tr_ID])
        @combine @astable begin
            :Sp_Int_Cent_Hes1 = mean(skipmissing(:Sp_Int_Cent_Hes1))
        end
        @rsubset !isnan(:Sp_Int_Cent_Hes1)
    end

    data_after_tmp = @rsubset(data_psm, :Differentiated == "yes")

    data_after = @chain data_after_tmp begin
        groupby(_, [:Region, :Medium, :Sp_Tr_ID])
        @combine @astable begin
            :Sp_Int_Cent_Hes1 = mean(skipmissing(:Sp_Int_Cent_Hes1))
        end
        @rsubset !isnan(:Sp_Int_Cent_Hes1)
    end

    fig = Figure(
        resolution=(350, 200),
        font="Helvetica Neue"
    )

    # Axes
    ax = Axis(
        fig[1, 1],
        ylabel="Hes1 (a.u.)",
        xticks=([1, 2, 3], ["PSM\nbefore diff.", "PSM\nafter diff.", "NT"]),
        xminorticksvisible=false,
        xminorgridvisible=false
    )

    # Plotting
    violin!(
        ax, repeat([1], size(data_before, 1)), data_before.Sp_Int_Cent_Hes1,
        datalimits=extrema, show_median=true,
        color=(mycol_default[1], 0.5), strokewidth=1, strokecolor=:black
    )
    violin!(
        ax, repeat([2], size(data_after, 1)), data_after.Sp_Int_Cent_Hes1,
        datalimits=extrema, show_median=true,
        color=(mycol_default[5], 0.5), strokewidth=1, strokecolor=:black
    )

    violin!(
        ax, repeat([3], size(data_nt, 1)), data_nt.Sp_Int_Cent_Hes1,
        datalimits=extrema, show_median=true,
        color=(mycol_default[2], 0.5), strokewidth=1, strokecolor=:black
    )

    y_pos = maximum(skipmissing([data_before.Sp_Int_Cent_Hes1; data_after.Sp_Int_Cent_Hes1; data_nt.Sp_Int_Cent_Hes1]))
    ax.limits = (nothing, nothing, 0, y_pos * 1.5)

    # Calculate p-values
    pval_before_after = pvalue(MannWhitneyUTest(data_before.Sp_Int_Cent_Hes1, data_after.Sp_Int_Cent_Hes1))
    pval_after_nt = pvalue(MannWhitneyUTest(data_after.Sp_Int_Cent_Hes1, data_nt.Sp_Int_Cent_Hes1))
    pval_before_nt = pvalue(MannWhitneyUTest(data_before.Sp_Int_Cent_Hes1, data_nt.Sp_Int_Cent_Hes1))

    # significance before and after diff
    bracket!(
        ax,
        1, y_pos,
        2, y_pos,
        text=_pvalue_to_string(pval_before_after),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    # significance after diff vs NT
    bracket!(
        ax,
        2, y_pos .* 1.1,
        3, y_pos .* 1.1,
        text=_pvalue_to_string(pval_after_nt),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    # significance before diff vs NT
    bracket!(
        ax,
        1, y_pos .* 1.3,
        3, y_pos .* 1.3,
        text=_pvalue_to_string(pval_before_nt),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    fig

end

# Save
fig_name = "Hes1-absolute_PSM-before-vs-after-differentiation_NT"
save(fig_path * fig_name * ".pdf", _plot_abs_hes1_before_vs_after_diff_w_NT())

# Save FWHM to txt file
function _csv_fwhm_abs_hes1_before_vs_after_diff_w_NT()
    
    # Collect data
    data_psm = @chain data_all_pp begin
        # Only PSM data in NB
        @rsubset :Region == "PSM" && :Medium == "NB"
        # Only tracks that have a differentiation point
        groupby(_, :Sp_Tr_ID)
        @transform :Diff_present = length(unique(:Differentiated)) == 2 ? "yes" : "no"
    end

    data_nt_tmp = @rsubset(data_all_pp, :Region == "NT" && :Medium == "NB")

    data_nt = @chain data_nt_tmp begin
        groupby(_, [:Region, :Medium, :Sp_Tr_ID])
        @combine @astable begin
            :Sp_Int_Cent_Hes1 = mean(skipmissing(:Sp_Int_Cent_Hes1))
        end
        @rsubset !isnan(:Sp_Int_Cent_Hes1)
    end

    data_before_tmp = @rsubset(data_psm, :Differentiated == "no")

    data_psm_before = @chain data_before_tmp begin
        groupby(_, [:Region, :Medium, :Sp_Tr_ID])
        @combine @astable begin
            :Sp_Int_Cent_Hes1 = mean(skipmissing(:Sp_Int_Cent_Hes1))
        end
        @rsubset !isnan(:Sp_Int_Cent_Hes1)
    end

    data_after_tmp = @rsubset(data_psm, :Differentiated == "yes")

    data_psm_after = @chain data_after_tmp begin
        groupby(_, [:Region, :Medium, :Sp_Tr_ID])
        @combine @astable begin
            :Sp_Int_Cent_Hes1 = mean(skipmissing(:Sp_Int_Cent_Hes1))
        end
        @rsubset !isnan(:Sp_Int_Cent_Hes1)
    end
    fwhm_df = DataFrame(
        Region = ["PSM"; "PSM"; "NT"],
        Differentiation = ["before"; "after"; "-"],
        FWHM_Hes1=[_fwhm(data_psm_before.Sp_Int_Cent_Hes1); _fwhm(data_psm_after.Sp_Int_Cent_Hes1); _fwhm(data_nt.Sp_Int_Cent_Hes1)])

    return fwhm_df
end

tab_name = "Hes1-absolute_PSM-before-vs-after-differentiation_NT_FWHM"
CSV.write(tab_path * tab_name * ".csv", _csv_fwhm_abs_hes1_before_vs_after_diff_w_NT())

#------------------------------------------------------------------------------
#
## PyBOATing individual tracks
#
#------------------------------------------------------------------------------

## PyBOAT setting basic parameters --------------------------------------------

# Sampling time (min)
T_sampling = 10.0

# Cut-off for the sinc smoothing (min) ~1.5*expected period
T_cutoff = 225.0

# Window size of amplitude envelope (min) ~3*expected period 
T_window = 450.0

# Period range (min)
period_start = 50.0
period_steps = 1.0
period_stop = 350.0
periods = collect(period_start:period_steps:period_stop)

# Wavelet power threshold for ridge detection (defined per region)
# 95th percentile
thresholds_df = DataFrame(
    Region = ["PSM"; "NT"; "PSM"; "NT"],
    Medium = ["NB"; "NB"; "ECM"; "ECM"],
    maxPower = [30.0; 40.0; 30.0; 40.0],
    p95 = [0.000; 0.000; 0.000; 0.000]
)

# Smoothing window size for ridge
window_ridge = 45

## Define autocorrelation fucntion --------------------------------------------
function _my_autocorrelation(
    signal::Vector{Float64};
    lags::Union{Vector{Int},Nothing}=nothing, confidence::Float64=0.95, peakwidth::Int=3)

    if lags === nothing
        lags = 1:1:Int(round(min(size(signal, 1) - 1, 10 * log10(size(signal, 1)))))
    end

    # Calculate autocorrelation with lags
    # Source: J. E. Hanke, D. W. Wichern, Business Forecasting (Pearson Education Limited, Essex, ed. 9, 2014).
    ac = autocor(signal, lags)

    # Degrees of freedom
    df = length(signal) - maximum(lags)

    # Critical value
    cv = quantile(TDist(df), (1 + confidence) / 2)

    # standard error
    se = [1 / sqrt(length(signal)); [sqrt((1 + 2 * sum(ac[1:(i-1)] .^ 2)) / length(signal)) for i in 2:maximum(lags)]]

    # confidence interval
    ci = cv .* se

    # Find peaks
    pks, vals = findmaxima(ac, peakwidth)
    pks_out = pks[vals.>ci[pks]]

    # Collect data into dataframe
    df = @chain DataFrame(
        AutoCorrCoeff=ac,
        Lags = collect(lags),
        ConfInt=[missing; ci]
        ) begin
            @rtransform :Peak_flag = :Lags ∈ [1; pks_out] ? true : false
        end

    return df

end


## Loop over tails and z-planes -----------------------------------------------
# Initialise output DataFrames
signals_df = DataFrame() # mean-normalised data -> pyboat only detrended
spectra_df = DataFrame()
fourier_df = DataFrame()
ridge_df = DataFrame()
autocor_df = DataFrame()

# Unique observations
unique_ids = unique(data_all_pp.Sp_Tr_ID)

# Loop over tails
for i in eachindex(unique_ids)

    # Non-differentiated data
    data_nondiff = @rsubset(data_all_pp, :Differentiated == "no")
    data_diff = data_all_pp

    # Extract data
    data_nondiff_i = @chain data_nondiff begin
        @rsubset :Sp_Tr_ID == unique_ids[i]
        dropmissing(_)
        @rtransform :Time_h = :Time_min ./ 60.0
        @orderby :Time_h
    end
    
    data_diff_i = @chain data_diff begin
        @rsubset :Sp_Tr_ID == unique_ids[i]
        dropmissing(_)
        @rtransform :Time_h = :Time_min ./ 60.0
        @orderby :Time_h
    end

    println("Track " * string(i) * "/" * string(length(unique_ids)))

    time_nondiff_i = data_nondiff_i.Time_min
    time_diff_i = data_diff_i.Time_min

    signal_nondiff_i = data_nondiff_i.Sp_Int_Cent_Hes1_Norm_Mean_RA
    signal_diff_i = data_diff_i.Sp_Int_Cent_Hes1_Norm_Mean_RA

    thresholds_i = @rsubset(thresholds_df, :Region == data_diff_i.Region[1] && :Medium == data_diff_i.Medium[1])

    pwr_thresh = thresholds_i.p95[1]
    pwr_max = thresholds_i.maxPower[1]

    # Period analysis of non-differentiated data only data (mean-normalised -> pyboat only detrended)
    signals_df_i, spectrum_df_i, fourier_df_i, ridge_df_i, fig_i = pyBoating(
        time_nondiff_i, signal_nondiff_i, "detrended", T_sampling, T_cutoff, T_window, periods, pwr_thresh, window_ridge, true, pwr_max
    )
    # Signals of all data (mean-normalised -> pyboat only detrended)
    signals_diff_df_i, _, _, _, _ = pyBoating(
        time_diff_i, signal_diff_i, "detrended", T_sampling, T_cutoff, T_window, periods, pwr_thresh, window_ridge, true, pwr_max
    )

    # Autocorrelation analysis
    ac_df_i = _my_autocorrelation(
        signals_df_i.detrended, lags=collect(0:Int(round(length(time_nondiff_i) / 2))),
        confidence=0.68)

    # Join signals with differentiation information
    data_diff_sel_i = rename(@select(data_diff_i, :Time_min, :Differentiated, :Diff_Time_min, :rel_Time_min), :Time_min => :time)
    signals_diff_df_i = outerjoin(signals_diff_df_i, data_diff_sel_i, on=:time)

    # ID df
    id_df = @select(DataFrame(data_nondiff_i[1, :]), :Medium, :Region, :Experiment, :Date, :Position, :Sp_Tr_ID)
    
    # Collect data
    global signals_df = [signals_df; [repeat(id_df, size(signals_diff_df_i, 1)) signals_diff_df_i]]
    global spectra_df = [spectra_df; [repeat(id_df, size(spectrum_df_i, 1)) spectrum_df_i]]
    global fourier_df = [fourier_df; [repeat(id_df, size(fourier_df_i, 1)) fourier_df_i]]
    global ridge_df = [ridge_df; [repeat(id_df, size(ridge_df_i, 1)) ridge_df_i]]
    global autocor_df = [autocor_df; [repeat(id_df, size(ac_df_i, 1)) ac_df_i]]

    # Save
    fig_path_alt = "../PS23_Figures/PS23_Single-cell-tracks/PS23_Spectra/PS23_Single-cell-tracks_"

    fig_norm_name = "Hes1-mean-norm-rollavg_detrended_period-spectrum_" * "exp-" * data_nondiff_i.Experiment[1] * "_track-" * replace(data_nondiff_i.Sp_Tr_ID[1], "_" => "-")
    save(fig_path_alt * fig_norm_name * ".pdf", fig_i)

end

# Find out dynamic range of power spectrum per tail and region
spectrum_dynrange = @chain spectra_df begin
    groupby(_, [:Region, :Medium])
    @combine $AsTable = (
        minPower=minimum(:power),
        maxPower=maximum(:power),
        sd=std(:power),
        sdx2=2 * std(:power),
        sdx3=3 * std(:power),
        p95=percentile(:power, 95),
        p97=percentile(:power, 97),
        p98=percentile(:power, 98),
        p99=percentile(:power, 99)
    )
end

# Representative data
signals_rep_df = DataFrame()
unique_ids = unique(data_rep_pp.Sp_Tr_ID)

# Loop over tails
for i in eachindex(unique_ids)

    # Extract data
    data_i = @chain data_rep_pp begin
        @rsubset :Sp_Tr_ID == unique_ids[i]
        dropmissing(_)
        @rtransform :Time_h = :Time_min ./ 60.0
        @orderby :Time_h
    end

    println("Track " * string(i) * "/" * string(length(unique_ids)))

    time_i = data_i.Time_h .* 60.0
    signal_i = data_i.Sp_Int_Cent_Hes1_Norm_Mean_RA

    thresholds_i = @rsubset(thresholds_df, :Region == data_i.Region[1] && :Medium == data_i.Medium[1])

    pwr_thresh = thresholds_i.p95[1]
    pwr_max = thresholds_i.maxPower[1]

    # pyboat with amplitude envelope
    signals_df_i, _, _, _, _ = pyBoating(
        time_i, signal_i, "detrended", T_sampling, T_cutoff, T_window, periods, pwr_thresh, window_ridge, true, pwr_max
    )

    # ID df
    id_df = @select(DataFrame(data_i[1, :]), :Medium, :Region, :Experiment, :Date, :Position, :Sp_Tr_ID)

    # Collect data
    global signals_rep_df = [signals_rep_df; [repeat(id_df, size(signals_df_i, 1)) signals_df_i]]
end

# Add differentation time point
diff_time = unique(@rsubset(data_rep_pp, :Region == "PSM").Diff_Time_min)[1]
signals_rep_df = @chain signals_rep_df begin
    @rtransform :Differentiated = :time >= diff_time && :Region == "PSM" ? "yes" : "no"
end


## Plot detrended w/ or w/o normalisation in individual facets ----------------
function _plot_pyboated_ind_tracks_facetted_nb_only()

    fig = Figure(
        resolution=(1200, 800),
        font="Helvetica Neue"
    )

    # Extract data
    data_plt = @chain signals_df begin
        @rsubset :Medium == "NB"
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
    end
    y_min = 0.9 * minimum(data_plt.detrended)
    y_max = 1.1 * maximum(data_plt.detrended)

    # Unique tracks
    unique_tracks = unique(data_plt.Tr_ID)
    n_trk = length(unique_tracks)

    # Define number of columns and rows
    n_col = 12
    n_row = n_trk ÷ n_col

    # Axes
    axs = [Axis(
        fig[fldmod1(i, n_col)...],
        xlabel="Time (h)",
        ylabel="Hes1 (a.u.)",
        xticks=0:6:24,
        xminorticks=0:1:24,
        limits=(-1, 25, y_min, y_max)
    ) for i in eachindex(unique_tracks)]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        # Extract data
        data_i = @chain data_plt begin
            @rsubset :Tr_ID == unique_tracks[i]
            @rtransform :Time_h = :time ./ 60.0
            @orderby :Time_h
        end

        # Axis
        ax.title = data_i.Region[1] * " (#" * string(data_i.Tr_ID[1]) * ")"

        if i ∉ 1:n_col:n_trk
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
        end
        if i ∉ (n_trk-n_col+1):n_trk
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        if data_i.Region[1] == "PSM"
            my_cols_i = mycol_default[1]
        elseif data_i.Region[1] == "NT"
            my_cols_i = mycol_default[2]
        end

        # Plot
        lines!(
            ax, data_i.Time_h, data_i.detrended,
            color=my_cols_i, linewidth=1)
        lines!(
            ax, data_i.Time_h, data_i.detrended,
            color=my_cols_i, linewidth=1)
    end

    if (n_trk % n_col) == 0
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row)]
    else
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row-1)]
        rowgap!(fig.layout, n_row, Relative(-0.07))
    end
    colgap!(fig.layout, Relative(0.005))

    fig

end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_ridge_individual-tracks_NB-only"
save(fig_path * fig_name * ".pdf", _plot_pyboated_ind_tracks_facetted_nb_only())


## Plot autocorrelations in individual facets ---------------------------------
function _plot_autocorrelations_ind_tracks_facetted_nb_only()

    fig = Figure(
        resolution=(1200, 800),
        font="Helvetica Neue"
    )

    # Extract data
    data_plt = @chain autocor_df begin
        @rsubset :Medium == "NB"
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
    end

    # Unique tracks
    unique_tracks = unique(data_plt.Tr_ID)
    n_trk = length(unique_tracks)

    # Define number of columns and rows
    n_col = 12
    n_row = n_trk ÷ n_col

    # Axes
    axs = [Axis(
        fig[fldmod1(i, n_col)...],
        xlabel="Lag time (h)",
        ylabel="ACC (a.u.)",
        xticks=0:2:6,
        limits=(-1, 6, -1, 1)
    ) for i in eachindex(unique_tracks)]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        # Extract data
        data_i = @chain data_plt begin
            dropmissing(_)
            @rsubset :Tr_ID == unique_tracks[i]
            @rtransform :Time_h = :Lags * 10.0 / 60.0
            @orderby :Time_h
        end

        # Axis
        ax.title = data_i.Region[1] * " (#" * string(data_i.Tr_ID[1]) * ")"

        if i ∉ 1:n_col:n_trk
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
        end
        if i ∉ (n_trk-n_col+1):n_trk
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        if data_i.Region[1] == "PSM"
            my_cols_i = mycol_default[1]
            my_bandcols_i = mycol_default[5]
        elseif data_i.Region[1] == "NT"
            my_cols_i = mycol_default[2]
            my_bandcols_i = mycol_default[7]
        end

        # Plot
        band!(
            ax, data_i.Time_h, data_i.ConfInt, -data_i.ConfInt,
            color=(my_bandcols_i, 0.5))
        stem!(
            ax, data_i.Time_h, data_i.AutoCorrCoeff, trunkcolor = my_cols_i,
            color=my_cols_i, stemcolor=my_cols_i, markersize = 5)
    end

    if (n_trk % n_col) == 0
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row)]
    else
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row-1)]
        rowgap!(fig.layout, n_row, Relative(-0.07))
    end
    colgap!(fig.layout, Relative(0.005))

    fig

end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_autocorrelation_individual-tracks_NB-only_68pc-confidence"
save(fig_path * fig_name * ".pdf", _plot_autocorrelations_ind_tracks_facetted_nb_only())


## Plot periods as estimated from the autocorrelation analysis ----------------
function _plot_autocorrelation_periods_nb_only()

    # Prepare plotting data
    data_plt = @chain autocor_df begin
        @rsubset :Medium == "NB" && :Peak_flag == true && :Lags != 1
        @rtransform :Period_h = (:Lags - 1) * 10.0 / 60.0
        @rsubset :Period_h < 6.0
        groupby(_, :Sp_Tr_ID)
        @combine $first
    end
    n_peaks = @combine(groupby(data_plt, :Region), :n = length(:Region)).n
    unique_region = unique(data_plt.Region)

    fig = Figure(
        resolution=(250, 200),
        font="Helvetica Neue"
    )

    ax = Axis(
        fig[1, 1],
        xticks=(1:2, unique_region .* "\n (n=" .* string.(n_peaks) .* ")"),
        ylabel="Period (h)",
        xminorticksvisible=false,
        xminorgridvisible=false,
    )

    # Loop over regions
    for i in eachindex(unique_region)

        # Extract data
        data_i = @rsubset(data_plt, :Region == unique_region[i])

        # Color
        if data_i.Region[1] == "PSM"
            mycol_i = mycol_default[1]
        elseif data_i.Region[1] == "NT"
            mycol_i = mycol_default[2]
        end

        violin!(
            ax, repeat([i], length(data_i.Period_h)), data_i.Period_h,
            show_median=true, datalimits=extrema,
            color=(mycol_i, 0.5), strokewidth=1, strokecolor=:black)

    end

    fig

end

fig_name = "Hes1-mean-norm-rollavg_detrended_autocorrelation-periods_violin-vs-region_NB-only_68pc-confidence"
save(fig_path * fig_name * ".pdf", _plot_autocorrelation_periods_nb_only())

## Plot fourier frequency plots in individual facets --------------------------
function _plot_autocorrelation_2rand_tracks_facetted_nb_only()

    fig = Figure(
        resolution=(400, 200),
        font="Helvetica Neue"
    )

    # Extract data
    data_plt = @rtransform(autocor_df, :Time_h = :Lags * 10.0 / 60.0)
    data_psm = @chain data_plt begin
        @rsubset :Medium == "NB" && :Region == "PSM" && !ismissing(:ConfInt)
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
    end
    data_psm = @subset(data_psm, :Tr_ID .== rand(1:maximum(data_psm.Tr_ID)))
    
    data_nt = @chain data_plt begin
        @rsubset :Medium == "NB" && :Region == "NT" && !ismissing(:ConfInt)
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
    end
    data_nt = @subset(data_nt, :Tr_ID .== rand(1:maximum(data_psm.Tr_ID)))

    # Axes
    axs = [Axis(
        fig[1, i],
        xlabel="Lag time (h)",
        ylabel="ACC (-)",
        xticks=0:2:6,
        xminorticks=0:1:6,
        limits=(0, 6, -1, 1)
    ) for i in 1:2]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        if i == 1
            ax.title = "PSM"
            my_cols_i = mycol_default[1]
            my_bandcols_i = mycol_default[5]
            band!(
                ax, data_psm.Time_h, data_psm.ConfInt, -data_psm.ConfInt,
                color=(my_bandcols_i, 0.5))
            stem!(
                ax, data_psm.Time_h, data_psm.AutoCorrCoeff, trunkcolor=mycol_default[1],
                color=mycol_default[1], stemcolor=mycol_default[1], markersize=7
                )
        elseif i == 2
            ax.title = "NT"
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
            my_cols_i = mycol_default[2]
            my_bandcols_i = mycol_default[7]
            band!(
                ax, data_nt.Time_h, data_nt.ConfInt, -data_nt.ConfInt,
                color=(my_bandcols_i, 0.5))
            stem!(
                ax, data_nt.Time_h, data_nt.AutoCorrCoeff, trunkcolor=mycol_default[2],
                color=mycol_default[2], stemcolor=mycol_default[2], markersize=7
                )
        end
    end

    colgap!(fig.layout, Relative(0.02))

    fig

end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_autocorrelation_2random-tracks_NB-only"
save(fig_path * fig_name * ".pdf", _plot_autocorrelation_2rand_tracks_facetted_nb_only())


## Plot fourier frequency plots in individual facets --------------------------
function _plot_fourier_freqs_ind_tracks_facetted_nb_only()

    fig = Figure(
        resolution=(1200, 800),
        font="Helvetica Neue"
    )

    # Extract data
    data_plt = @chain fourier_df begin
        @rsubset :Medium == "NB" && :frequency > 0
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
    end
    y_max = 1.1 * maximum(data_plt.power)

    # Unique tracks
    unique_tracks = unique(data_plt.Tr_ID)
    n_trk = length(unique_tracks)

    # Define number of columns and rows
    n_col = 12
    n_row = n_trk ÷ n_col

    # Axes
    axs = [Axis(
        fig[fldmod1(i, n_col)...],
        xlabel="Period (h)",
        ylabel="Power (-)",
        xticks=0:6:24,
        xminorticks=0:1:24,
        limits=(-1, 24, -1, y_max)
    ) for i in eachindex(unique_tracks)]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        # Extract data
        data_i = @chain data_plt begin
            @rsubset :Tr_ID == unique_tracks[i]
            @rtransform :Period_h = 1.0 / :frequency / 60.0
        end

        # Axis
        ax.title = data_i.Region[1] * " (#" * string(data_i.Tr_ID[1]) * ")"

        if i ∉ 1:n_col:n_trk
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
        end
        if i ∉ (n_trk-n_col+1):n_trk
            ax.xlabelvisible = false
            ax.xticklabelsvisible = false
        end

        if data_i.Region[1] == "PSM"
            my_cols_i = mycol_default[1]
        elseif data_i.Region[1] == "NT"
            my_cols_i = mycol_default[2]
        end

        # Plot
        stem!(
            ax, data_i.Period_h, data_i.power, trunkcolor=my_cols_i,
            color=my_cols_i, stemcolor=my_cols_i, markersize=5)
    end

    if (n_trk % n_col) == 0
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row)]
    else
        [rowgap!(fig.layout, i, Relative(0.01)) for i in 1:(n_row-1)]
        rowgap!(fig.layout, n_row, Relative(-0.07))
    end
    colgap!(fig.layout, Relative(0.005))

    fig

end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_fourier-frequencies_individual-tracks_NB-only"
save(fig_path * fig_name * ".pdf", _plot_fourier_freqs_ind_tracks_facetted_nb_only())


## Plot fourier frequency plots in individual facets --------------------------
function _plot_fourier_freqs_2rand_tracks_facetted_nb_only()

    fig = Figure(
        resolution=(400, 200),
        font="Helvetica Neue"
    )

    # Extract data
    data_psm = @chain fourier_df begin
        @rsubset :Medium == "NB" && :frequency > 0 && :Region == "PSM"
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
        @rtransform :Period_h = 1.0 / :frequency / 60.0
    end
    data_psm = @subset(data_psm, :Tr_ID .== rand(1:maximum(data_psm.Tr_ID)))
    data_nt = @chain fourier_df begin
        @rsubset :Medium == "NB" && :frequency > 0 && :Region == "NT"
        # Unique track ID
        groupby(_, :Sp_Tr_ID)
        @transform :Tr_ID = $groupindices
        @rtransform :Period_h = 1.0 / :frequency / 60.0
    end
    data_nt = @subset(data_nt, :Tr_ID .== rand(1:maximum(data_psm.Tr_ID)))

    y_max = 1.1 * maximum([maximum(data_psm.power), maximum(data_nt.power)])

    # Axes
    axs = [Axis(
        fig[1, i],
        xlabel="Period (h)",
        ylabel="Power (-)",
        xticks=0:2:8,
        xminorticks=0:1:8,
        limits=(0, 8, -1, y_max)
    ) for i in 1:2]

    # Loop over tracks
    for (i, ax) in enumerate(axs)

        if i == 1
            ax.title = "PSM"
            stem!(
                ax, data_psm.Period_h, data_psm.power, trunkcolor=mycol_default[1],
                color=mycol_default[1], stemcolor=mycol_default[1], markersize=7
                )
        elseif i == 2
            ax.title = "NT"
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
            stem!(
                ax, data_nt.Period_h, data_nt.power, trunkcolor=mycol_default[2],
                color=mycol_default[2], stemcolor=mycol_default[2], markersize=7
                )
        end
    end

    colgap!(fig.layout, Relative(0.02))

    fig

end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_fourier-frequencies_2random-tracks_NB-only"
save(fig_path * fig_name * ".pdf", _plot_fourier_freqs_2rand_tracks_facetted_nb_only())

## Plot period as estimated by fourier ----------------------------------------
function _plt_fourier_period_nb_only()
    
    # Prepare plotting data
    data_plt = @chain fourier_df begin
        @rsubset :Medium == "NB" && :power > percentile(fourier_df.power, 98)
        @rtransform :Period_h = 1.0 / :frequency / 60.0
    end
    unique_region = unique(data_plt.Region)

    fig = Figure(
        resolution=(250, 200),
        font="Helvetica Neue"
    )

    # Prepare data
    data_psm = @rsubset(data_plt, :Region == "PSM")
    data_nt = @rsubset(data_plt, :Region == "NT")

    y_pos = maximum(skipmissing([data_psm[:, :Period_h]; data_nt[:, :Period_h]]))
    
    ax = Axis(
        fig[1, 1],
        xticks=(1:2, unique_region),
        yticks=1:1:6,
        ylabel="Period (h)",
        xminorticksvisible=false,
        xminorgridvisible=false,
        limits=(nothing, nothing, 1, 5.5)
    )

    

    violin!(
        ax, repeat([1], length(data_psm[:, :Period_h])), data_psm[:, :Period_h],
        show_median=true, datalimits=extrema,
        color=(mycol_default[1], 0.5), strokewidth=1, strokecolor=:black)

    violin!(
        ax, repeat([2], length(data_nt[:, :Period_h])), data_nt[:, :Period_h],
        show_median=true, datalimits=extrema,
        color=(mycol_default[2], 0.5), strokewidth=1, strokecolor=:black)

    # Calculate p-values
    pval_psm_nt = pvalue(MannWhitneyUTest(Float64.(data_psm[:, :Period_h]), Float64.(data_nt[:, :Period_h])))

    # significance PSM vs NT
    bracket!(
        ax,
        1, y_pos * 1.1,
        2, y_pos * 1.1,
        text=_pvalue_to_string(pval_psm_nt),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)


    fig

end

fig_name = "Hes1-mean-norm-rollavg_detrended_fourier-periods_violin-vs-region_NB-only"
save(fig_path * fig_name * ".pdf", _plt_fourier_period_nb_only())


## Plot periods/amplitudes of PSM/NT in NB -------------------------------------------
function _plot_ridge_period_amplitude_violin(per_amp::String)

    fig = Figure(
        resolution=(250, 200),
        font="Helvetica Neue"
    )

    # Make plot dataset
    data_tmp = @chain ridge_df begin
        @rsubset :Medium == "NB" && !ismissing(:time)
        @rtransform :periods = :periods / 60
    end
    unique_region = unique(data_tmp.Region)

    Time_frame = (1:120)

    data_plt = @chain data_tmp begin
        @rsubset :time ∈ (Time_frame *10)
        groupby(_, [:Medium, :Region, :Experiment, :Date, :Position, :Sp_Tr_ID])
        @combine @astable begin
            :amplitude = mean(skipmissing(:amplitude))
            :periods = mean(skipmissing(:periods))
        end
        @rsubset !isnan(:amplitude)
        @rsubset !isnan(:periods)
    end

    # Prepare data
    data_psm = @rsubset(data_plt, :Region == "PSM")
    data_nt = @rsubset(data_plt, :Region == "NT")
    
    y_pos = maximum(skipmissing([data_psm[:, per_amp]; data_nt[:, per_amp]]))
    
    if per_amp == "periods"
        my_ylabel = "Period (h)"
        my_lims = (nothing, nothing, 1, y_pos * 1.2)
        my_yticks=1:1:6
    elseif per_amp == "amplitude"
        my_ylabel = "Amplitude (a.u.)"
        my_lims = (nothing, nothing, 0, y_pos * 1.2)
        my_yticks=0:0.1:0.4
    end

    ax = Axis(
        fig[1, 1],
        xticks=(1:2, unique_region),
        ylabel=my_ylabel,
        yticks=my_yticks,
        xminorticksvisible=false,
        xminorgridvisible=false,
        limits = my_lims
    )

    violin!(
        ax, repeat([1], length(data_psm[:, per_amp])), data_psm[:, per_amp],
        show_median=true, datalimits=extrema,
        color=(mycol_default[1], 0.5), strokewidth=1, strokecolor=:black)

    violin!(
        ax, repeat([2], length(data_nt[:, per_amp])), data_nt[:, per_amp],
        show_median=true, datalimits=extrema,
        color=(mycol_default[2], 0.5), strokewidth=1, strokecolor=:black)

    # Calculate p-values
    pval_psm_nt = pvalue(MannWhitneyUTest(Float64.(data_psm[:, per_amp]), Float64.(data_nt[:, per_amp])))

    # significance PSM vs NT
    bracket!(
        ax,
        1, y_pos * 1.1,
        2, y_pos * 1.1,
        text=_pvalue_to_string(pval_psm_nt),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    fig
end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_ridge_individual-tracks_period-violin-vs-region_NB"
save(fig_path * fig_name * ".pdf", _plot_ridge_period_amplitude_violin("periods"))

fig_name = "Hes1-mean-norm-rollavg_detrended_ridge_individual-tracks_amplitude-violin-vs-region_NB"
save(fig_path * fig_name * ".pdf", _plot_ridge_period_amplitude_violin("amplitude"))


## Plot periods/amplitudes of PSM/NT in NB -------------------------------------------
function _plot_spectra_power_violin()

    fig = Figure(
        resolution=(250, 200),
        font="Helvetica Neue"
    )

    # Make plot dataset
    data_tmp = @chain spectra_df begin
        @rsubset :Medium == "NB" && !ismissing(:time)
    end
    unique_region = unique(data_tmp.Region)

    Time_frame = (1:120)

    data_plt = @chain data_tmp begin
        @rsubset :time ∈ (Time_frame *10)
        groupby(_, [:Medium, :Region, :Experiment, :Date, :Position, :Sp_Tr_ID])
        @combine @astable begin
            :power = mean(skipmissing(:power))
        end
        @rsubset !isnan(:power)
    end

    # Prepare data
    data_psm = @rsubset(data_plt, :Region == "PSM")
    data_nt = @rsubset(data_plt, :Region == "NT")
    
    y_pos = maximum(skipmissing([data_psm[:, :power]; data_nt[:, :power]]))
    
   
    my_ylabel = "Wavelet Power"
    my_lims = (nothing, nothing, 1, y_pos * 1.2)
    my_yticks=1:2:10
    

    ax = Axis(
        fig[1, 1],
        xticks=(1:2, unique_region),
        ylabel=my_ylabel,
        yticks=my_yticks,
        xminorticksvisible=false,
        xminorgridvisible=false,
        limits = my_lims
    )

    violin!(
        ax, repeat([1], length(data_psm[:, :power])), data_psm[:, :power],
        show_median=true, datalimits=extrema,
        color=(mycol_default[1], 0.5), strokewidth=1, strokecolor=:black)

    violin!(
        ax, repeat([2], length(data_nt[:, :power])), data_nt[:, :power],
        show_median=true, datalimits=extrema,
        color=(mycol_default[2], 0.5), strokewidth=1, strokecolor=:black)

    # Calculate p-values
    pval_psm_nt = pvalue(MannWhitneyUTest(Float64.(data_psm[:, :power]), Float64.(data_nt[:, :power])))

    # significance PSM vs NT
    bracket!(
        ax,
        1, y_pos * 1.1,
        2, y_pos * 1.1,
        text=_pvalue_to_string(pval_psm_nt),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    fig
end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_spectra_individual-tracks_power-violin-vs-region_NB"
save(fig_path * fig_name * ".pdf", _plot_spectra_power_violin())




# Save FWHM to txt file
function _csv_fwhm_ridge_period(per_amp::String)

    # Calculate
    fwhm_df_tmp = @chain ridge_df begin
        @rsubset :Medium == "NB" && !ismissing(:time)
        groupby(_, :Region)
    end
    if per_amp == "periods"
        fwhm_df = @combine(fwhm_df_tmp, :FWHM_Periods_h = _fwhm(Float64.(:periods)) ./ 60)
    elseif per_amp == "amplitude"
        fwhm_df = @combine(fwhm_df_tmp, :FWHM_Amplitude = _fwhm(Float64.(:amplitude)))
    end

    return fwhm_df
end

tab_name = "Hes1-mean-norm-rollavg_detrended_ridge_individual-tracks_period-FWHM-vs-region_NB"
CSV.write(tab_path * tab_name * ".csv", _csv_fwhm_ridge_period("periods"))

tab_name = "Hes1-mean-norm-rollavg_detrended_ridge_individual-tracks_amplitude-FWHM-vs-region_NB"
CSV.write(tab_path * tab_name * ".csv", _csv_fwhm_ridge_period("amplitude"))


## Plot detrended representative tracks in PSM and NT --------------
function _plot_pyboat_signal_hes1_repr(region::String)

    # Extract data
    data_plt = @orderby(@rsubset(signals_rep_df, :Region == region), :time)

    # Figure and Axes
    fig = Figure(
        resolution=(250, 200),
        font="Helvetica Neue"
    )

    ax = Axis(
        fig[1, 1],
        xlabel="Time (h)",
        ylabel="Hes1 (a.u.)",
        xticks=0:6:24,
        limits=(0, 22, -0.5, 0.5)
    )

    # Color
    if region == "PSM"
        data_i_nodiff = @rsubset(data_plt, :Differentiated == "no")
        data_i_yesdiff = @rsubset(data_plt, :Differentiated == "yes")
        lines!(
            ax, data_i_nodiff.time ./ 60.0, data_i_nodiff.detrended,
            color= mycol_default[1])
        lines!(
            ax, data_i_yesdiff.time ./ 60.0, data_i_yesdiff.detrended,
            color= mycol_default[5])
    elseif region == "NT"
        lines!(
            ax, data_plt.time ./ 60.0, data_plt.detrended,
            color=mycol_default[2])
    end

    fig

end

# Save
fig_name = "Hes1-mean-norm-rollavg_detrended_representative-PSM"
save(fig_path * fig_name * ".pdf", _plot_pyboat_signal_hes1_repr("PSM"))

fig_name = "Hes1-mean-norm-rollavg_detrended_representative-NT"
save(fig_path * fig_name * ".pdf", _plot_pyboat_signal_hes1_repr("NT"))

## Plot period comparison wavelet vs fourier vs autocorrelation ---------------
function _plot_period_comparison_wvlt_fr_ac(region::String)
    data_wavelet = @chain ridge_df begin
        @rsubset :Medium == "NB" && :Region == region && !ismissing(:periods)
        groupby(_, [:Medium, :Experiment, :Date, :Position, :Sp_Tr_ID])
        @combine @astable begin
            :periods = mean(skipmissing(:periods))
        end
        @rsubset !isnan(:periods)
        @rtransform :Period_h = :periods / 60.0
    end
    
    data_fourier = @chain fourier_df begin
        @rsubset :Medium == "NB" && :Region == region && :frequency > 0 && !ismissing(:frequency) && :power > percentile(fourier_df.power, 98)
        @rtransform :Period_h = 1 ./ :frequency / 60.0
    end
    data_autocor = @chain autocor_df begin
        @rsubset :Medium == "NB" && :Region == region && :Peak_flag == true && :Lags != 1 && !ismissing(:Lags)
        @rtransform :Period_h = (:Lags - 1) * 10.0 / 60.0
    end

    fig = Figure(
        resolution=(350, 200),
        font="Helvetica Neue"
    )

    ax = Axis(
        fig[1, 1],
        title=region,
        xticks=(1:3, ["Wavelet", "Fourier", "Autocorrelation"]),
        # yticks=0:2:9,
        ylabel="Period (h)",
        xminorticksvisible=false,
        xminorgridvisible=false
    )

    violin!(
        ax, repeat([1], length(data_wavelet.Period_h)), data_wavelet.Period_h,
        show_median=true, datalimits=extrema,
        color=(mycol_default[1], 0.5), strokewidth=1, strokecolor=:black)
    violin!(
        ax, repeat([2], length(data_fourier.Period_h)), data_fourier.Period_h,
        show_median=true, datalimits=extrema,
        color=(mycol_default[2], 0.5), strokewidth=1, strokecolor=:black)
    violin!(
        ax, repeat([3], length(data_autocor.Period_h)), data_autocor.Period_h,
        show_median=true, datalimits=extrema,
        color=(mycol_default[3], 0.5), strokewidth=1, strokecolor=:black)

    y_pos = maximum(skipmissing([data_wavelet.Period_h; data_fourier.Period_h; data_autocor.Period_h]))
    ax.limits = (nothing, nothing, nothing, y_pos * 1.5)

    # Calculate p-values
    pval_wvlt_four = pvalue(MannWhitneyUTest(data_wavelet.Period_h, data_fourier.Period_h))
    pval_four_ac = pvalue(MannWhitneyUTest(data_fourier.Period_h, data_autocor.Period_h))
    pval_wvlt_ac = pvalue(MannWhitneyUTest(data_wavelet.Period_h, data_autocor.Period_h))

    # significance Wavelet - Fourier
    bracket!(
        ax,
        1, y_pos * 1.05,
        2, y_pos * 1.05,
        text=_pvalue_to_string(pval_wvlt_four),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    # significance Fourier -  Autocorrelation
    bracket!(
        ax,
        2, y_pos .* 1.15,
        3, y_pos .* 1.15,
        text=_pvalue_to_string(pval_four_ac),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    # significance Wavelt - Autocorrelation
    bracket!(
        ax,
        1, y_pos .* 1.3,
        3, y_pos .* 1.3,
        text=_pvalue_to_string(pval_wvlt_ac),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    fig

end

fig_name = "Hes1-mean-norm-rollavg_detrended_PSM_NB_Period-Wavelet-Fourier-Autocorr"
save(fig_path * fig_name * ".pdf", _plot_period_comparison_wvlt_fr_ac("PSM"))

fig_name = "Hes1-mean-norm-rollavg_detrended_NT_NB_Period-Wavelet-Fourier-Autocorr"
save(fig_path * fig_name * ".pdf", _plot_period_comparison_wvlt_fr_ac("NT"))



## Plot period comparison wavelet vs fourier ---------------
function _plot_period_comparison_wvlt_fr(region::String)
    data_wavelet = @chain ridge_df begin
        @rsubset :Medium == "NB" && :Region == region && !ismissing(:periods)
        groupby(_, [:Medium, :Experiment, :Date, :Position, :Sp_Tr_ID])
        @combine @astable begin
            :periods = mean(skipmissing(:periods))
        end
        @rsubset !isnan(:periods)
        @rtransform :Period_h = :periods / 60.0
    end
    
    data_fourier = @chain fourier_df begin
        @rsubset :Medium == "NB" && :Region == region && :frequency > 0 && !ismissing(:frequency) && :power > percentile(fourier_df.power, 98)
        @rtransform :Period_h = 1 ./ :frequency / 60.0
    end
    data_autocor = @chain autocor_df begin
        @rsubset :Medium == "NB" && :Region == region && :Peak_flag == true && :Lags != 1 && !ismissing(:Lags)
        @rtransform :Period_h = (:Lags - 1) * 10.0 / 60.0
    end

    fig = Figure(
        resolution=(250, 200),
        font="Helvetica Neue"
    )

    ax = Axis(
        fig[1, 1],
        title=region,
        xticks=(1:2, ["Wavelet", "Fourier"]),
        # yticks=0:2:9,
        ylabel="Period (h)",
        xminorticksvisible=false,
        xminorgridvisible=false,
        limits=(nothing, nothing, 1, 5.5)
    )

    violin!(
        ax, repeat([1], length(data_wavelet.Period_h)), data_wavelet.Period_h,
        show_median=true, datalimits=extrema,
        color=(mycol_default[1], 0.5), strokewidth=1, strokecolor=:black)
    violin!(
        ax, repeat([2], length(data_fourier.Period_h)), data_fourier.Period_h,
        show_median=true, datalimits=extrema,
        color=(mycol_default[2], 0.5), strokewidth=1, strokecolor=:black)

    y_pos = maximum(skipmissing([data_wavelet.Period_h; data_fourier.Period_h]))
    ax.limits = (nothing, nothing, 1, y_pos * 1.5)

    # Calculate p-values
    pval_wvlt_four = pvalue(MannWhitneyUTest(data_wavelet.Period_h, data_fourier.Period_h))
    
    # significance Wavelet - Fourier
    bracket!(
        ax,
        1, y_pos * 1.05,
        2, y_pos * 1.05,
        text=_pvalue_to_string(pval_wvlt_four),
        textoffset=6, style=:square, fontsize=12, linewidth=1, width=2)

    
    fig

end

fig_name = "Hes1-mean-norm-rollavg_detrended_PSM_NB_Period-Wavelet-Fourier"
save(fig_path * fig_name * ".pdf", _plot_period_comparison_wvlt_fr("PSM"))

fig_name = "Hes1-mean-norm-rollavg_detrended_NT_NB_Period-Wavelet-Fourier"
save(fig_path * fig_name * ".pdf", _plot_period_comparison_wvlt_fr("NT"))



########################
## Plot coefficient of variation based on FT---------------
function _csv_CV_period_fourier()
    # Prepare plotting data using fourier data
    data_plt = @chain fourier_df begin
        @rsubset :Medium == "NB" && :power > percentile(fourier_df.power, 98)
        @rtransform :periods = 1.0 / :frequency / 60.0
    end
    data_fourier = @chain data_plt begin
        groupby(_, [:Medium, :Region])
        @combine @astable begin
            :cov = std(skipmissing(:periods))/mean(skipmissing(:periods))
        end
        @rsubset !isnan(:cov)
    end

   
    
end

tab_name = "Hes1-mean-norm-rollavg_detrended_NB_Period-Fourier_COV"
CSV.write(tab_path * tab_name * ".csv", _csv_CV_period_fourier())

function _csv_CV_period_wavelet()
    # Prepare plotting data using fourier data
    data_wavelet = @chain ridge_df begin
        @rsubset :Medium == "NB" && !ismissing(:periods)
        groupby(_, [:Medium, :Region])
        @combine @astable begin
            :cov = std(skipmissing(:periods))/mean(skipmissing(:periods))
        end
        @rsubset !isnan(:cov)
    end
   
end

tab_name = "Hes1-mean-norm-rollavg_detrended_NB_Period-wavelet_COV"
CSV.write(tab_path * tab_name * ".csv", _csv_CV_period_wavelet())





function _csv_fwhm_period_comparison_wvlt_fr_ac(region::String)
    fwhm_wavelet = @chain ridge_df begin
        @rsubset :Medium == "NB" && :Region == region && !ismissing(:periods)
        groupby(_, [:Medium, :Experiment, :Date, :Position, :Sp_Tr_ID])
        @combine @astable begin
            :periods = mean(skipmissing(:periods))
        end
        @rsubset !isnan(:periods)
        @combine :FWHM_Periods_h = _fwhm(Float64.(:periods)) ./ 60
    end
    fwhm_fourier = @chain fourier_df begin
        @rsubset :Medium == "NB" && :Region == region && :frequency > 0 && !ismissing(:frequency) && :power > percentile(fourier_df.power, 98)
        @combine :FWHM_Periods_h = _fwhm(Float64.(1 ./ :frequency)) ./ 60
    end
    fwhm_autocor = @chain autocor_df begin
        @rsubset :Medium == "NB" && :Region == region && :Peak_flag == true && :Lags != 1 && !ismissing(:Lags)
        @rtransform :Period_h = (:Lags - 1) * 10.0 / 60.0
        @combine :FWHM_Periods_h = _fwhm(Float64.(:Period_h)) ./ 60
    end

    fwhm_df = DataFrame(
        Method = ["Wavelet"; "Fourier"; "Autocorrelation"],
        FWHM_Period_h=[fwhm_wavelet[1, 1], fwhm_fourier[1, 1], fwhm_autocor[1, 1]]
    )

    return fwhm_df

end

tab_name = "Hes1-mean-norm-rollavg_detrended_PSM_NB_Period-Wavelet-Fourier-Autocorr_FWHM"
CSV.write(tab_path * tab_name * ".csv", _csv_fwhm_period_comparison_wvlt_fr_ac("PSM"))

tab_name = "Hes1-mean-norm-rollavg_detrended_NT_NB_Period-Wavelet-Fourier-Autocorr_FWHM"
CSV.write(tab_path * tab_name * ".csv", _csv_fwhm_period_comparison_wvlt_fr_ac("NT"))


function _plot_pyboated_signal_heatmap(region::String; time_sel::String, sort::String)

    # Prepare data
    data = @chain signals_df begin
        @rsubset :Region == region && :Medium == "NB"
        groupby(_, :Sp_Tr_ID)
        @transform :min_Time = time_sel == "absolute" ? minimum(:time) : minimum(:rel_Time_min)
        @rtransform :Time = time_sel == "absolute" ? :time : :rel_Time_min
    end

    fig = Figure(
        resolution=(300, 400),
        font="Helvetica Neue"
    )

    ax = Axis(
        fig[1, 1],
        xlabel=time_sel == "absolute" ? "Time (h)" : "Rel. time (h)",
        ylabel="Tracks",
        xticks=-24:6:24,
        xgridvisible=false,
        ygridvisible=false,
        xminorgridvisible=false,
        yminorgridvisible=false,
        xminorticksvisible=false,
        yminorticksvisible=false,
        yticklabelsvisible=false,
        yticksvisible=false,
        leftspinevisible=false,
        # limits=(-1, 25, nothing, nothing)
    )

    if sort == "t0"
        data_plt = @chain data begin
            @orderby(_, :min_Time, :Time)
            groupby(_, :Sp_Tr_ID)
            @transform :Tr_ID = $groupindices
        end
    elseif sort == "t0diff"
        data_plt = @chain data begin
            @orderby(_, :min_Time, :Diff_Time_min, :Time)
            groupby(_, :Sp_Tr_ID)
            @transform :Tr_ID = $groupindices
        end
    else
        data_plt = @chain data begin
            groupby(_, :Sp_Tr_ID)
            @transform :Tr_ID = $groupindices
        end
    end
    
    if region == "PSM"
        hm = heatmap!(
            ax, data_plt.Time ./ 60.0, data_plt.Tr_ID, data_plt.detrended,
            colormap = reverse(cgrad(:vik)), 
            lowclip=nothing, highclip=nothing,
            colorrange=(-1, 1))

        # define colorbar ticks
        cb_ticks = -1:0.5:1

        # Differentiation times
        if time_sel == "absolute"
            data_diff = @rsubset(data_plt, :Time == :Diff_Time_min)
            scatter!(
                ax, data_diff.Diff_Time_min ./ 60.0, data_diff.Tr_ID, 
                color=:black, markersize=5
                )
        else
            scatter!(
                ax, repeat([0.0], length(unique(data_plt.Tr_ID))), unique(data_plt.Tr_ID),
                color=:black, markersize=5
                )
        end
    elseif region == "NT"
        hm = heatmap!(
            ax, data_plt.Time ./ 60.0, data_plt.Tr_ID, data_plt.detrended,
            colormap = reverse(cgrad(:vik)), 
            lowclip=nothing, highclip=nothing,
            colorrange=(-0.5, 0.5))

        # define colorbar ticks
        cb_ticks = -0.5:0.5:0.5
        
    end

    Colorbar(
        fig[:, end+1], hm, label="Norm. Hes1 (a.u.)", ticks=cb_ticks,
        height=100, width=10, tellheight=false
        )
    colgap!(fig.layout, Relative(0.02))

    fig
end

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_PSM_abs-time_unsort"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("PSM", time_sel="absolute", sort="none"))

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_PSM_abs-time_sort-t0"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("PSM", time_sel="absolute", sort="t0"))

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_PSM_abs-time_sort-t0-diff"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("PSM", time_sel="absolute", sort="t0diff"))

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_PSM_rel-time_unsort"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("PSM", time_sel="relative", sort="none"))

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_PSM_rel-time_sort-t0"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("PSM", time_sel="relative", sort="t0"))

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_NT_abs-time_unsort"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("NT", time_sel="absolute", sort="none"))

fig_name = "Hes1-mean-norm-rollavg_detrended_tracks-heatmap_NB-only_NT_abs-time_sort"
save(fig_path * fig_name * ".pdf", _plot_pyboated_signal_heatmap("NT", time_sel="absolute", sort="t0"))
