"""
atto_gui.py

A lightweight GUI that reads raw spectra appended by `main.py` to a rolling buffer file
and displays the averaged spectrum updated every 2 seconds (average over last 5 raw
acquisitions). It also shows a small terminal view with the last 20 lines printed by
`main.py` (captured to a rotating log file called `atto_last_lines.log`).

This GUI is intentionally simple and non-blocking: it reads files written by the
acquisition thread/process and updates the plot on a 2s timer. It does not attempt to
control acquisition.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
from matplotlib import patches

# Config
ROLLING_AVG_COUNT = 5
UPDATE_INTERVAL_S = 0.4
LOG_LINES = 20

# File locations (these should match what main.py writes)
BUF_FILE = os.path.join(os.path.dirname(__file__), 'atto_recent.npy')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'atto_last_lines.log')
ENERGY_FILE = os.path.join(os.path.dirname(__file__), 'Spec.txt')


def read_buffer():
    """Read the recent raw spectra buffer saved by main.py.
    Expected format: 2D numpy array shape (N, M) with dtype uint16.
    Returns None on failure.
    """
    try:
        if not os.path.exists(BUF_FILE):
            return None
        arr = np.load(BUF_FILE)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def read_log_lines(n=LOG_LINES):
    try:
        if not os.path.exists(LOG_FILE):
            return []
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
        return lines[-n:]
    except Exception:
        return []


def main():
    # Interactive mode on
    plt.ion()

    # Create a 1920x1080 window (figsize in inches; use dpi=100)
    fig = plt.figure(figsize=(19.2, 10.8), facecolor='#0f0f0f')
    # Make both panels equal height (50% each) and leave small spacing
    gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

    ax_spec = fig.add_subplot(gs[0])
    ax_term = fig.add_subplot(gs[1])

    # Try to load energy axis from Spec.txt; fall back to pixel indices
    energy_eV = None
    energy_len = 1340
    try:
        if os.path.exists(ENERGY_FILE):
            energy_eV = np.loadtxt(ENERGY_FILE)
            energy_len = energy_eV.size
    except Exception:
        energy_eV = None

    # Spectrum plot styling
    ax_spec.set_title('Live averaged spectrum (updates every 2s)', color='white', pad=12, loc='left', fontsize=16)
    ax_spec.set_xlabel('Energy (eV)' if energy_eV is not None else 'Pixel', color='#dddddd', fontsize=12)
    ax_spec.set_ylabel('Counts', color='#dddddd', fontsize=12)
    ax_spec.set_facecolor('#0b0b0b')
    ax_spec.tick_params(colors='#cccccc')
    for spine in ax_spec.spines.values():
        spine.set_color('#2b2b2b')
    ax_spec.grid(True, color='#1f1f1f', linestyle='--', linewidth=0.6)

    # Initial empty plot (x length will be adapted when data appears)
    if energy_eV is not None:
        x = energy_eV.copy()
    else:
        x = np.arange(energy_len)
    line, = ax_spec.plot(x, np.zeros_like(x), color='#00d1ff', linewidth=2.0)
    fill = ax_spec.fill_between(x, 0, 0, color='#00d1ff', alpha=0.08)

    # Terminal area styling
    ax_term.set_facecolor('#0b0b0b')
    ax_term.axis('off')

    # Add a subtle box around the terminal area
    # Slightly inset the box to provide a margin for the text and avoid overlapping UI chrome
    # Make the box almost fill the terminal axes while keeping a tiny margin at bottom
    box = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle='round,pad=0.02',
        transform=ax_term.transAxes,
        linewidth=1.0,
        edgecolor='#00ff66',
        facecolor='#070707',
        zorder=0
    )
    ax_term.add_patch(box)

    # Terminal will render each line as its own Text artist so we can color lines individually
    term_font = {'family': 'Consolas', 'size': 12}
    term_artists = []

    # Stop button: create a small axes in the top-right corner of the figure
    stop_ax = fig.add_axes([0.88, 0.92, 0.09, 0.06])
    stop_ax.set_axis_off()
    stop_button_rect = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.1', transform=stop_ax.transAxes,
                                              facecolor='#ff3b30', edgecolor='#880000')
    stop_ax.add_patch(stop_button_rect)
    stop_label = stop_ax.text(0.5, 0.5, 'STOP', va='center', ha='center', fontsize=14, fontweight='bold', color='white', transform=stop_ax.transAxes)

    stop_flag_file = os.path.join(os.path.dirname(__file__), 'atto_stop.flag')
    stopped_flag_file = os.path.join(os.path.dirname(__file__), 'atto_stopped.flag')

    def on_stop_clicked():
        try:
            with open(stop_flag_file, 'w', encoding='utf-8') as f:
                f.write('stop')
            print('GUI requested stop (stop flag written).')
        except Exception as e:
            print(f'Could not write stop flag: {e}')

    # Register mouse click on the stop area using a proper handler
    def _mouse_click_handler(ev):
        if ev.inaxes == stop_ax:
            on_stop_clicked()

    fig.canvas.mpl_connect('button_press_event', _mouse_click_handler)

    plt.tight_layout()

    last_update = 0

    while plt.fignum_exists(fig.number):
        t = time.time()
        # If main process signalled completion by writing the stopped flag, exit GUI
        if os.path.exists(stopped_flag_file):
            try:
                plt.close(fig)
                break
            except Exception:
                break
        if t - last_update >= UPDATE_INTERVAL_S:
            buf = read_buffer()
            if buf is not None and buf.size > 0:
                # take last ROLLING_AVG_COUNT raw acquisitions
                recent = buf[-ROLLING_AVG_COUNT:]
                avg = recent.mean(axis=0)
                # update x if energy axis matches, otherwise use pixel index
                if energy_eV is not None and energy_eV.size == avg.size:
                    x = energy_eV
                else:
                    x = np.arange(avg.size)
                line.set_xdata(x)
                line.set_ydata(avg)
                # update fill_between by removing old and adding new
                try:
                    # remove previous collection
                    for coll in ax_spec.collections:
                        coll.remove()
                except Exception:
                    pass
                ax_spec.fill_between(x, 0, avg, color='#00d1ff', alpha=0.14)

                ax_spec.set_ylim(0, max(np.max(avg) * 1.1, 1000))
                if energy_eV is not None and energy_eV.size == avg.size:
                    ax_spec.set_xlim(np.min(x), np.max(x))
                else:
                    ax_spec.set_xlim(0, avg.size)

            # update terminal lines (render each line separately so we can color them)
            lines = read_log_lines()
            display_lines = list(lines[-LOG_LINES:]) if lines else []

            # If user pressed Stop (stop flag exists) show waiting message appended
            if os.path.exists(stop_flag_file):
                display_lines.append('')
                display_lines.append('Shutdown requested, waiting for acquisition to finish...')

            # remove previous text artists
            try:
                for a in term_artists:
                    try:
                        a.remove()
                    except Exception:
                        pass
            except Exception:
                pass
            term_artists.clear()

            # vertical spacing in axes coordinates
            top_y = 0.8
            line_height = 0.035
            y = top_y
            for ln in display_lines[-LOG_LINES:]:
                text_color = '#999999'  # default grey
                low = ln.lower()
                if 'flush' in low or 'server' in low or 'local' in low:
                    text_color = '#00ff66'  # green for flush messages
                if 'violation' in low or 'acquisition time violation' in low:
                    text_color = '#ff4444'  # red for violations

                artist = ax_term.text(0.02, y, ln, va='top', ha='left', fontfamily='Consolas', fontsize=12, color=text_color, transform=ax_term.transAxes, zorder=10)
                term_artists.append(artist)
                y -= line_height

            fig.canvas.draw_idle()
            last_update = t

        plt.pause(0.1)

if __name__ == '__main__':
    main()
