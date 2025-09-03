import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from tqdm import tqdm
import os
from matplotlib import gridspec
#from atas_metrics import SpectrumMetrics
from typing import Optional, Tuple
# imoort polynomial fitting functions
from numpy.polynomial.polynomial import polyfit, polyval

# for plotting:
import sys
sys.path.append('/home/marcel/Koofr/University/Code/moritz code')
# noinspection PyUnresolvedReferences
#from common_functions import FigSettings  # Import class
#from common_functions import *


class DataLabel(IntEnum):
    OFF = 0
    ON = 1
    TRAINING = 2
    DISCARDED = 3


###################################################################################################################
# ATAS ON/OFF PLOTTING FUNCTION ##############################################################
###################################################################################################################
def atas_on_off_clas(xuv_spectra: np.ndarray, xuv_energies_eV: np.ndarray, identifiers: np.ndarray,
                     shift: int = -1, delay_stepsize_mm = -1, flip: bool = False,
                     save_path_thesis: str = "atas_spectrum.pdf", extract_line_fs: int = 0,
                     save_path: str = None) -> np.ndarray:
    """
    Plots the ON/OFF spectrum classically

    :param xuv_spectra: 2D array of XUV spectra.
    :param xuv_energies_eV: 1D array of XUV energies in eV.
    :param identifiers: 2D array of identifiers indicating: 1) the type of the spectrum according to DataLabel,
                        2) the delaystep number 3) the block number.
    :param shift: If > 0, shifts the indices of the spectra by this value.
    :param delay_stepsize_mm: If provided, the delay steps will be converted to femtoseconds using this value.
                           If -1, the delay steps will be treated as indices.
    :param flip: If True, flips the ATAS OD values for plotting.
    :param save_path_thesis: Path to save the figure, if provided.
    :param save_path: If provided, saves the figure at this path.
    :param extract_line_fs: If > 0, extracts a single line at this delay in femtoseconds.
                            If 0, no line is extracted.
    :return: 2D ATAS spectrum array with shape (n_delays, n_energy).
    """

    # plot the summed up XUV spectra:
    # summed_up_spectra = np.sum(xuv_spectra, axis=1)  # Sum over the first axis (spectra)
    # plt.figure(figsize=(10, 6))
    # plt.plot(summed_up_spectra, label="Summed XUV Spectra", color='blue')
    # plt.xlabel("Index")
    # plt.ylabel("Intensity [a.u.]")
    # plt.title("Summed XUV Spectra")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # first count the number of different delay steps and blocks
    delay_steps = np.unique(identifiers[:, 1])
    blocks = np.unique(identifiers[:, 2])
    spds = len(blocks) - 1
    print(f"Found {len(delay_steps)} delay steps and {spds} on-off blocks in the identifiers.")

    # calculate the spss
    _, counts = np.unique(identifiers[:, 2], return_counts=True)
    values, value_counts = np.unique(counts, return_counts=True)

    # initialize the ATAS 2D array for the ON and OFF spectra
    atas_spectrum = np.zeros((len(delay_steps), len(xuv_energies_eV)), dtype=np.float64)

    # set up the ON and OFF spectra (use as SUMs)
    off_spectrum = np.zeros(len(xuv_energies_eV), dtype=np.float64)
    on_spectrum = np.zeros(len(xuv_energies_eV), dtype=np.float64)

    # NEW: counters for unequal counts
    on_count = 0
    off_count = 0

    # calculate the mean off spectrum (unchanged)
    mean_off_spectrum = np.mean(xuv_spectra[identifiers[:, 0] == DataLabel.OFF], axis=0)

    eps = 1e-12  # NEW: guard to avoid log(0)/division-by-zero

    # loop over the delay steps and blocks
    for delay_step in delay_steps:
        indices = np.where(identifiers[:, 1] == delay_step)[0]

        spds_counter = 0
        # NEW: reset accumulators per delay step
        on_spectrum.fill(0.0); off_spectrum.fill(0.0)
        on_count = 0; off_count = 0

        for idx in indices:
            # apply a shift ONCE (fixes the double-shift bug)
            idx = idx + shift
            if idx >= len(xuv_spectra) or idx < 0:
                continue

            # accumulate sums and counts
            if identifiers[idx, 0] == DataLabel.ON:
                on_spectrum += xuv_spectra[idx]  # <- no + shift here
                on_count += 1
            elif identifiers[idx, 0] == DataLabel.OFF:
                off_spectrum += xuv_spectra[idx]
                off_count += 1
            else:
                continue

            # if current is OFF and the next is not OFF, close and compute OD
            if (idx + 1 < len(xuv_spectra)
                    and identifiers[idx, 0] == DataLabel.OFF
                    and identifiers[idx + 1, 0] != DataLabel.OFF):

                if on_count > 0 and off_count > 0:
                    on_mean = on_spectrum / on_count
                    off_mean = off_spectrum / off_count

                    # OD = ln(ON/OFF), with guards for zeros/negatives
                    OD_values = np.log(
                        np.maximum(on_mean, eps) / np.maximum(off_mean, eps)
                    )

                    # accumulate into ATAS spectrum (keep your normalization by spds)
                    atas_spectrum[int(delay_step), :] += OD_values / spds

                # reset for the next pair within this delay step
                on_spectrum.fill(0.0); off_spectrum.fill(0.0)
                on_count = 0; off_count = 0

                spds_counter += 1

        # print(f"Finished delay step {delay_step} with {spds_counter} blocks.")

    # flip the ATAS spectrum if needed
    if flip:
        atas_spectrum = atas_spectrum * -1

    # calculate the timestep axis for the x-axis of the plot
    if delay_stepsize_mm > 0:
        timestep_fs = distance_mm_to_delay_fs(delay_stepsize_mm)
        delay_steps_fs = np.arange(len(delay_steps)) * timestep_fs
        # x_label = rf"Pump-Probe Delay $\tau$ [fs], $\Delta \tau \approx ${int(timestep_fs)} fs"
        x_label = rf"Pump-Probe Delay [fs]"
    else:
        delay_steps_fs = np.arange(len(delay_steps))
        x_label = "Delay Step"

    # Plot the ATAS spectrum
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(delay_steps_fs, xuv_energies_eV)
    vmax = np.abs(atas_spectrum).max()
    vmin = -vmax
    # cmap = plt.get_cmap('seismic')
    cmap = plt.get_cmap('bwr')  # Blue-White-Red colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # The pcm object is what we'll make interactive
    pcm = plt.pcolormesh(X, Y, atas_spectrum.T, shading='auto', cmap=cmap, norm=norm)
    # rasterize the pcolormesh for better performance and saving
    pcm.set_rasterized(True)

    plt.xlabel(x_label)
    plt.ylabel("XUV Energy [eV]")
    plt.colorbar(pcm, label=rf"Pump-Induced Signal [$\Delta$mOD]")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


    # PLOT EXTRACTED LINES:
    # --- Find index closest to 28 eV ---
    # target_energy = 28.0  # eV
    # idx_28eV = np.argmin(np.abs(xuv_energies_eV - target_energy))

    # # --- Extract the line ---
    # line_at_28eV = atas_spectrum[:, idx_28eV]

    # # --- Plot the extracted line ---
    # plt.figure(figsize=(10, 4))
    # plt.plot(delay_steps_fs, line_at_28eV, label=f"{target_energy} eV")
    # plt.xlabel("Delay (fs)")
    # plt.ylabel(rf"Pump-Induced Signal [$\Delta$mOD]")
    # plt.title(f"ATAS Lineout at {target_energy} eV")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # extract the line at the specified time if requested
    # extracted_line, picked_time_fs, idx = extract_atas_line_at_time(
    #     atas_spectrum, delay_steps_fs, extract_line_fs
    # )

    # If a specific line is requested, the helper below can be used by callers separately.
    # We always return the full 2D ATAS spectrum for downstream processing (e.g., MATLAB).
    return atas_spectrum


def plot_on_off_clas_shift_sweep(
    xuv_spectra: np.ndarray,
    xuv_energies_eV: np.ndarray,
    identifiers: np.ndarray,
    delay_stepsize_mm: float = -1.0,
    flip: bool = False,
    use_background_correction: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Sweep the ON/OFF classical OD calculation over integer shifts in [-4, 4] using
    the *same logic* as in `plot_on_off_clas` (including the double-shift on spectra
    access), and plot a 3x3 grid of results.

    :param xuv_spectra: 2D array (n_samples, n_energy) of XUV spectra.
    :param xuv_energies_eV: 1D array (n_energy,) of XUV energies in eV.
    :param identifiers: 2D int array (n_samples, 3): [DataLabel, delay_step, block].
    :param delay_stepsize_mm: If > 0, converts delay steps to femtoseconds via `distance_mm_to_delay_fs`.
    :param flip: If True, multiplies OD values by -1 (sign flip).
    :param use_background_correction: If True, subtracts the 50th percentile per delay row (axis=1).
    :param save_path: If provided, saves the 3×3 figure at this path.
    :return: None
    """
    # --- Delay axis like in your original ---
    delay_steps = np.unique(identifiers[:, 1])
    blocks = np.unique(identifiers[:, 2])
    spds = len(blocks) - 1  # exactly like your code
    n_energy = xuv_energies_eV.shape[0]

    if delay_stepsize_mm > 0:
        timestep_fs = distance_mm_to_delay_fs(delay_stepsize_mm)
        delay_axis = np.arange(len(delay_steps)) * timestep_fs
        x_label = r"Pump-Probe Delay [fs]"
    else:
        delay_axis = np.arange(len(delay_steps))
        x_label = "Delay Step"

    eps = 1e-12  # numerical guard for log

    # === Helper that reproduces your original block logic verbatim (including double shift on spectra) ===
    def compute_atas_spectrum_for_shift(shift: int) -> np.ndarray:
        atas_spectrum = np.zeros((len(delay_steps), n_energy), dtype=np.float64)
        on_spectrum = np.zeros(n_energy, dtype=np.float64)
        off_spectrum = np.zeros(n_energy, dtype=np.float64)

        for dsi, delay_step in enumerate(delay_steps):
            indices = np.where(identifiers[:, 1] == delay_step)[0]

            # reset accumulators for each delay step like in your code
            on_spectrum[:] = 0.0
            off_spectrum[:] = 0.0
            spds_counter = 0

            for idx0 in indices:
                # apply a shift if needed (exactly as in your function)
                idx = idx0 + shift
                if idx < 0 or idx >= len(xuv_spectra):
                    continue

                # === Your original uses identifiers[idx, 0] to decide ON/OFF,
                #     but accumulates xuv_spectra[idx + shift] (double shift) ===
                x_idx = idx + shift
                if x_idx < 0 or x_idx >= len(xuv_spectra):
                    # safety guard to avoid IndexError at the edges for larger shifts
                    continue

                if identifiers[idx, 0] == DataLabel.ON:
                    on_spectrum += xuv_spectra[x_idx]
                elif identifiers[idx, 0] == DataLabel.OFF:
                    off_spectrum += xuv_spectra[x_idx]
                else:
                    continue

                # end-of-OFF-block detection (exactly like your code)
                if (idx + 1) < len(xuv_spectra) and identifiers[idx, 0] == DataLabel.OFF and identifiers[idx + 1, 0] != DataLabel.OFF:
                    ratio = (on_spectrum + eps) / (off_spectrum + eps)
                    OD_values = np.log(ratio)

                    atas_spectrum[int(delay_step), :] += OD_values / max(spds, 1)
                    # reset accumulators
                    on_spectrum[:] = 0.0
                    off_spectrum[:] = 0.0
                    spds_counter += 1

        if flip:
            atas_spectrum *= -1.0

        # mandatory 50th percentile unless disabled
        if use_background_correction:
            atas_spectrum = atas_spectrum - np.percentile(atas_spectrum, 50, axis=1, keepdims=True)

        return atas_spectrum

    # --- Compute all 9 shifts with exactly the same logic ---
    shifts = list(range(-4, 5))  # [-4, ..., +4]
    atas_list = [compute_atas_spectrum_for_shift(s) for s in shifts]

    # consistent color scaling across subplots
    all_vals = np.stack(atas_list, axis=0)
    vmax = float(np.nanmax(np.abs(all_vals)))
    vmin = -vmax
    cmap = plt.get_cmap('bwr')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- Plot 3x3 grid ---
    X, Y = np.meshgrid(delay_axis, xuv_energies_eV)
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    pcm_last = None
    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            s = shifts[i * 3 + j]
            pcm_last = ax.pcolormesh(X, Y, atas_list[i * 3 + j].T, shading='auto', cmap=cmap, norm=norm)
            ax.set_title(f"shift = {s}", fontsize=10)
            if i == 2:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel("XUV Energy [eV]")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(pcm_last, cax=cbar_ax, label=rf"Pump-Induced Signal [$\Delta$mOD]")

    bc_note = "with 50th-percentile background" if use_background_correction else "without background correction"
    fig.suptitle(f"ATAS (ON/OFF, log ratio) shift sweep (−4…+4), {bc_note}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def extract_atas_line_at_time(
    atas_spectrum: np.ndarray,
    delay_steps_fs: np.ndarray,
    extract_line_fs: float
) -> Tuple[np.ndarray, float, int]:
    """
    Extracts the 1D ATAS spectrum line closest to a given delay time.

    :param atas_spectrum: 2D array shaped (n_delays, n_energies).
    :param delay_steps_fs: 1D array of delays in fs (length n_delays).
    :param extract_line_fs: Target time in fs.
    :return: Tuple containing:
        - line_1d (np.ndarray): The extracted spectrum line.
        - picked_time_fs (float): The actual delay time used.
        - picked_index (int): The index of the extracted line.
    """
    delay_steps_fs = np.asarray(delay_steps_fs, dtype=float)
    idx = int(np.argmin(np.abs(delay_steps_fs - extract_line_fs)))
    return atas_spectrum[idx, :], float(delay_steps_fs[idx]), idx


####################################################################################################
# Functions for miscellaneous tasks ################################################################
####################################################################################################
def distance_mm_to_delay_fs(distance_mm: float) -> float:
    """
    Convert delay stage travel distance to optical delay in femtoseconds.

    :param distance_mm: Distance moved by the delay stage in millimeters.
    :return: Corresponding time delay in femtoseconds.
    """
    c_mm_per_ns = 299.792458  # speed of light in mm/ns
    delay_fs = (2 * distance_mm / c_mm_per_ns) * 1e6  # factor 2 for round-trip, ns -> fs
    return delay_fs