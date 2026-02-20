import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import spikeglx
from powerpixels import Pipeline

this_path = Path(__file__).parent



def get_ni_pulse_times(rec_path):
    sync_times = np.load(rec_path / 'raw_ephys_data' / '_spikeglx_sync.times.npy')
    sync_polarities = np.load(rec_path / 'raw_ephys_data' / '_spikeglx_sync.polarities.npy')
    sync_channels = np.load(rec_path / 'raw_ephys_data' / '_spikeglx_sync.channels.npy')
    with open(rec_path / 'raw_ephys_data' / '_spikeglx_sync.pinout.json') as f:
        sync_map = json.load(f)

    pulse_times = {}
    for ch_name, pin in sync_map.items():
        if ch_name == 'imec_sync':
            continue
        onsets = sync_times[(sync_channels == pin) & (sync_polarities == 1)]
        offsets = sync_times[(sync_channels == pin) & (sync_polarities == -1)]
        if len(onsets) > 0 and len(offsets) > 0:
            onsets = onsets[onsets < offsets[-1]]
            offsets = offsets[offsets > onsets[0]]
        ch_pulse = {
            'onset': onsets,
            'offset': offsets
        }
        pulse_times[ch_name] = ch_pulse

    return pulse_times


def analyze_time_chunk(rec_slice):
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
    peaks = detect_peaks(
        rec_slice,  method='locally_exclusive',
        method_kwargs=dict(
            detect_threshold=5,
            radius_um=50.
        ),
        job_kwargs=job_kwargs
    )
    peak_locations = localize_peaks(
        rec_slice, peaks, method='center_of_mass', job_kwargs=job_kwargs
    )
    return peaks, peak_locations


def main(rec_path: Path):
    pp = Pipeline(this_path.parent/'config'/'threshold_settings.json', rec_path)

    # Detect data format
    pp.detect_data_format()

    # Initialize NIDAQ synchronization
    if pp.settings['USE_NIDAQ']:
        pp.set_nidq_paths()
        pp.extract_sync_pulses()
        pp.extract_stim_pulses()
    
    pulse_times = get_ni_pulse_times(rec_path)

    # Loop over multiple probes
    raw_probes = spikeglx.get_probes_from_folder(pp.session_path)
    for i, this_probe in enumerate(raw_probes):
        print(f'\nStarting preprocessing of {this_probe}')

        # Set probe paths
        pp.set_probe_paths(this_probe)

        # Decompress raw data if necessary
        print(f'\nDecompressing {this_probe}')
        pp.decompress()

        print(f'\nLoading {this_probe} recording')
        # Load in raw data
        rec = pp.load_raw_binary()

        # Preprocessing
        rec = pp.preprocessing(rec)

        chan_frs = {}
        baseline_frs = {}
        chan_spikes = {}
        baseline_spikes = {}
        chan_locs = {}
        baseline_locs = {}
        for chan, onoff in pulse_times.items():
            n_onoff = len(onoff['onset'])
            print(f"\nAnalyzing {n_onoff} {chan}")
            ch_peaks, ch_locs, ch_frs = [], [], []
            bl_peaks, bl_locs, bl_frs = [], [], []
            if len(onoff['onset']) == 0:
                continue
            for n, (onset, offset) in enumerate(zip(onoff['onset'], onoff['offset'])):
                print(f"\nAnalyzing {n+1}/{n_onoff} {chan}")
                if onset < 1:
                    continue
                if offset - onset < 1.0:
                    offset = onset + 1.0
                onset = onset + rec.get_start_time()
                offset = offset + rec.get_start_time()
                baseline_ch = rec.time_slice(start_time=onset-1.0, end_time=onset)
                bl_peaks_i, bl_locs_i = analyze_time_chunk(baseline_ch)
                rec_ch = rec.time_slice(start_time=onset, end_time=offset)
                ch_peaks_i, ch_locs_i = analyze_time_chunk(rec_ch)
                if len(ch_peaks_i):
                    ch_peaks.append(ch_peaks_i)
                    ch_locs.append(ch_locs_i)
                if len(bl_peaks_i):
                    bl_peaks.append(bl_peaks_i)
                    bl_locs.append(bl_locs_i)
                ch_frs.append(len(ch_peaks_i) / (offset - onset))
                bl_frs.append(len(bl_peaks_i) / (onset - (onset - 1)))
                if n>1:
                    break
            if len(ch_peaks):
                chan_spikes[chan] = np.concatenate(ch_peaks, axis=0)
                chan_locs[chan] = np.concatenate(ch_locs, axis=0)
            chan_frs[chan] = ch_frs
            if len(bl_peaks):
                baseline_spikes[chan] = np.concatenate(bl_peaks, axis=0)
                baseline_locs[chan] = np.concatenate(bl_locs, axis=0)
            baseline_frs[chan] = bl_frs

        xorder = []
        fr_df = {'chan': [], 'fr': []}
        for chan, frs in chan_frs.items():
            xorder.extend([chan+"_baseline", chan])
            fr_df['chan'].extend([chan]*len(frs))
            fr_df['fr'].extend(frs)
            bl_frs = baseline_frs[chan]
            fr_df['chan'].extend([chan+"_baseline"]*len(bl_frs))
            fr_df['fr'].extend(bl_frs)
        fr_df = pd.DataFrame.from_dict(fr_df)
        print(fr_df.head())
        ax = sns.barplot(fr_df, x='chan', y='fr', order=xorder)
        ax.get_figure().savefig(pp.results_path / 'pp_barplot')

        fig, axs = plt.subplots(2, len(pulse_times), sharex=True, sharey=True)
        for ax in axs.flat:
            si.plot_probe_map(rec, ax=ax, with_channel_ids=False)
        for j, (chan, locs) in enumerate(chan_locs.items()):
            axs[0, j].scatter(locs['x'], locs['y'], color='red', alpha=0.01)
            axs[0, j].set_xlabel('')
            axs[0, j].set_ylabel('')
            axs[0, j].set_title(f'{chan} spikes')
        for j, (chan, locs) in enumerate(baseline_locs.items()):
            axs[1, j].scatter(locs['x'], locs['y'], color='red', alpha=0.01)
            axs[1, j].set_xlabel('')
            axs[1, j].set_ylabel('')
        axs[0, 0].set_ylabel('TTL On')
        axs[1, 0].set_ylabel('Pre-Baseline')
        axs[0, 0].set_ylim([-100, 5000])
        fig.savefig(pp.results_path / 'pp_map')
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main(Path(r"C:/Users/ixb240017/Box/Neuropixels_Sharing/CATGT/20260218_ABATE125/LC_g3_t0"))
