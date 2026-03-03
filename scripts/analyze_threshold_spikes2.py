import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import spikeinterface.full as si
from spikeinterface.preprocessing import compute_motion, load_motion_info
# from spikeinterface.sortingcomponents.peak_detection import detect_peaks
# from spikeinterface.sortingcomponents.motion import estimate_motion
# from spikeinterface.sortingcomponents.peak_localization import localize_peaks
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
        else:
            continue
        ch_pulse = {
            'onset': onsets,
            'offset': offsets
        }
        pulse_times[ch_name] = ch_pulse

    return pulse_times


def main(rec_path: Path, psth_range=(-1.0, 1.0)):
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

        if not (pp.results_path / 'threshold_motion').exists():
            # Preprocessing
            rec = pp.preprocessing(rec)

            job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
            # peaks = detect_peaks(
            #     rec,  method='locally_exclusive',
            #     method_kwargs=dict(
            #         detect_threshold=5,
            #         radius_um=50.
            #     ),
            #     job_kwargs=job_kwargs
            # )
            # peak_locations = localize_peaks(
            #     rec, peaks, method='center_of_mass', job_kwargs=job_kwargs
            # )
            motion, motion_info = compute_motion(
                recording=rec,
                preset='dredge_fast',
                folder=pp.results_path / 'threshold_motion',
                output_motion_info=True,
                detect_kwargs=dict(
                    method='locally_exclusive',
                    detect_threshold=5,
                    radius_um=50.,
                ),
                localize_peaks_kwargs=dict(method='center_of_mass'),
                **job_kwargs
            )
            # np.save(pp.results_path / 'threshold_peaks.npy', peaks)
            # np.save(pp.results_path / 'threshold_locations.npy', peak_locations)
        else:
            motion_info = load_motion_info(pp.results_path / 'threshold_motion')

        fs = rec.sampling_frequency
        fig = plt.figure(figsize=(14, 8))
        si.plot_motion_info(
            motion_info, rec,
            figure=fig,
            color_amplitude=True,
            amplitude_cmap="inferno",
            scatter_decimate=10,
        )
        fig.savefig(pp.results_path / 'threshold_motion.png')
        times = motion_info['peaks']['sample_index'] / fs
        channels = motion_info['peaks']['channel_index']
        labels = np.zeros(times.shape, dtype=int)
        locs_x = motion_info['peak_locations']['x']
        locs_y = motion_info['peak_locations']['y']

        sec_bins = np.arange(min(times), max(times), 1.0)
        millisec_bins = np.arange(min(times), max(times), 0.1)
        millisec_labels = np.zeros(millisec_bins.shape, dtype=int)

        fig, ax = plt.subplots()
        ax.hist(times, bins=sec_bins)
        ch_peaks = {}
        for i, (chan, onoff) in enumerate(pulse_times.items()):
            for j, (on, off) in enumerate(zip(onoff['onset'], onoff['offset'])):
                if j == 0:
                    label = chan
                else:
                    label = None
                ax.axvspan(on, off, color=colors[i], alpha=0.3, label=label)
                labels[np.logical_and(times > on, times < off)] = i+1
                millisec_labels[np.logical_and(millisec_bins > on, millisec_bins < off)] = i+1
        ax.legend()

        fig, axs = plt.subplots(len(pulse_times), 1)
        psth_dict = {}
        for i, (chan, onoff) in enumerate(pulse_times.items()):
            baseline_times = []
            psth_times = []
            for onset in onoff['onset']:
                baseline_times.append(times[np.logical_and(times > onset + psth_range[0], times < onset)] - onset)
                psth_times.append(times[np.logical_and(times >= onset, times < onset + psth_range[1])] - onset)
            baseline_times = np.hstack(baseline_times)
            psth_times = np.hstack(psth_times)
            axs[i].hist(baseline_times, bins=np.arange(psth_range[0], 0.1, 0.1), color='gray')
            axs[i].hist(psth_times, bins=np.arange(0, psth_range[1]+0.1, 0.1), color=colors[i])
        plt.tight_layout()
        fig.savefig(pp.results_path / 'threshold_psth')

        u_chan = np.unique(motion_info['peaks']['channel_index'])
        fig, axs = plt.subplots(1, len(pulse_times), sharex=True, sharey=True)
        for i, chan in enumerate(pulse_times.keys()):
            responder_inds = np.zeros(times.shape, dtype=bool)
            for ch in u_chan:
                bl_times = times[np.logical_and(channels == ch, labels==0)]
                ch_times = times[np.logical_and(channels == ch, labels==i+1)]
                bl_fr = len(bl_times) / (np.sum(millisec_labels == 0) * 0.1)
                ch_fr = len(ch_times) / (np.sum(millisec_labels == i+1) * 0.1)
                if ch_fr > bl_fr:
                    responder_inds = np.logical_or(responder_inds, np.logical_and(channels == ch, labels==i+1))
            si.plot_probe_map(rec, ax=axs[i], with_channel_ids=False)
            axs[i].scatter(locs_x[responder_inds], locs_y[responder_inds], color='red', alpha=0.01)
        fig.savefig(pp.results_path / 'threshold_map')


        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main(
            Path(
                r"C:\Users\ianba\Box\Neuropixels_Sharing\CATGT\20260227_LCP02\LC_g0_t1"
            )
        )
