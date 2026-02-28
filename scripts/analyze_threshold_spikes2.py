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


this_path = Path(__file__)
PPP_path = this_path.parent.parent


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
    pp = Pipeline(
        PPP_path / 'config' / 'threshold_settings.json',
        rec_path
    )

    # Detect data format
    pp.detect_data_format()
    
    colors = sns.color_palette('Set2', n_colors=6)

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
        
        if not (pp.results_path / 'threshold_peaks.npy').exists():
            # Preprocessing
            rec = pp.preprocessing(rec)
            
            job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
            peaks = detect_peaks(
                rec,  method='locally_exclusive',
                method_kwargs=dict(
                    detect_threshold=5,
                    radius_um=50.
                ),
                job_kwargs=job_kwargs
            )
            peak_locations = localize_peaks(
                rec, peaks, method='center_of_mass', job_kwargs=job_kwargs
            )
            np.save(pp.results_path / 'threshold_peaks.npy', peaks)
            np.save(pp.results_path / 'threshold_locations.npy', peak_locations)
        else:
            peaks = np.load(pp.results_path / 'threshold_peaks.npy')
            peak_locations = np.load(pp.results_path / 'threshold_locations.npy')
            
        fs = rec.sampling_frequency
        times = peaks['sample_index'] / fs
        
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)
        ax.scatter(peaks['sample_index'] / fs, peak_locations['y'], color='k', marker='.', alpha=0.002)
        
        ch_peaks = {}
        for i, (chan, onoff) in enumerate(pulse_times.items()):
            ch_peaks[chan] = []
            for on, off in zip(onoff['onset'], onoff['offset']):
                ax.axvspan(on, off, color=colors[i], alpha=0.3)
                ch_peaks[chan].append(peaks[np.logical_and(peaks['sample_index']/fs>on, peaks['sample_index']/fs < off)])
        
        fig, ax = plt.subplots(1,1)
        ax.hist(peaks['sample_index']/fs, bins=round(np.max(peaks['sample_index']) / fs // 0.1))
        for i, (chan, onoff) in enumerate(pulse_times.items()):
            for on, off in zip(onoff['onset'], onoff['offset']):
                ax.axvspan(on, off, color=colors[i], alpha=0.3)
        
        fig, ax = plt.subplots(len(ch_peaks),1)
        for i, (chan, spike_list) in enumerate(ch_peaks.items()):
            chan_spikes = []
            for spikes, onset in zip(spike_list, pulse_times[chan]['onset']):
                chan_spikes.append(spikes['sample_index'] / fs - onset)
            chan_spikes = np.hstack(chan_spikes)
            ax.hist(chan_spikes, bins=np.arange(min(chan_spikes), max(chan_spikes), 0.1))
            ax.set_title(chan)
        plt.show()
        
        


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main(
            Path(
                r"C:/Users/ixb240017/Box/Neuropixels_Sharing/CATGT/20260227_LCP02/LC_g0_t1"
            )
        )
