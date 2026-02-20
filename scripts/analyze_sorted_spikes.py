import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import windows, convolve

import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import spikeglx
from powerpixels import Pipeline, load_neural_data


this_path = Path(__file__).parent


def filter_onoff(onsets, offsets):
    if len(offsets) > 0:
        onsets = onsets[onsets < offsets[-1]]
    else:
        onsets = np.array([])
    if len(onsets) > 0:
        offsets = offsets[offsets > onsets[0]]
    else:
        offsets = np.array([])


def load_trig_times(alf_path):
    di_channels = [
        'jaw_tap',
        'paw_tap',
        'tail_tap'
    ]
    di_colors = sns.color_palette('tab10', n_colors=len(di_channels), as_cmap=False)
    di = {}
    for dich, diclr in zip(di_channels, di_colors):
        on = np.load(alf_path / dich+'_onset_times.npy')
        off = np.load(alf_path / dich+'_offset_times.npy')
        di[dich]['onsets'], di[dich]['offsets'] = filter_onoff(on, off)
        di[dich]['color'] = diclr
        filter_onoff(di[dich])
    
    on = np.load(alf_path / 'vns_train_onset_times.npy')
    off = np.load(alf_path / 'vns_train_offset_times.npy')
    trains = dict(zip(['onsets', 'offsets'], filter_onoff(on, off)))
    trains['color'] = sns.color_palette('Set2', n_colors=len(trains['onsets']), as_cmap=False)
    
    pulses = {
        'times': np.load(alf_path / 'vns_pulse.times.npy'),
        'train': np.load(alf_path / 'vns_pulse.train.npy'),
        'i_amps': np.load(alf_path / 'vns_current.amps.npy'),
        'v_amps': np.load(alf_path / 'vns_voltage.amps.npy'),
    }
    return di, trains, pulses


def compute_fr(spikes, u_clust, bins):
    frs = np.zeros(len(u_clust), len(bins) - 1)
    for ci in u_clust:
        c_idx = spikes['clusters'] == ci
        frs[i] = np.histogram(spikes['times'][c_idx], bins=bins)
    return u_clust, frs


def smooth_fr(hist_fr, dt, sigma=50.0, axis=-1):
    kern = windows.gaussian((round(sigma / dt) * 6) + 1, sigma)
    k_shape = np.ones(len(hist_fr.shape))
    k_shape[axis] = len(kern)
    kern = np.reshape(kern, k_shape)
    smooth_fr = convolve(hist_fr, kern, method='same')
    return smooth_fr


def plot_raster(
    a
):
    pass


def plot_raster_psth(
    spikes,
    times,
    pre_ms = 300,
    post_ms = 800,
    bin_sz = 50,
    color='gray',
    ttl='',
    fig=None,
    markers=[],
    marker_kws={
        'color': 'k'
    },
):
    if fig is None:
        fig, axs = plt.subplots(2,1, sharex=True, sharey=True)
    else:
        axs = fig.add_subplot(2,1, sharex=True, sharey=True)
    
    pass
    
    bins = np.arange(-pre_ms, post_ms + bin_sz, bin_sz)
    centers = bins[:-1] + np.diff(bins)
    
    for 

    return axs

def run(rec_path: Path):
    print(f"Starting {rec_path}")
        
    pp = Pipeline(this_path.parent / 'config' / 'dummy_settings.json')
    raw_probes = spikeglx.get_probes_from_folder(pp.session_path)
    for i, this_probe in enumerate(raw_probes):
        print(f'\nStarting preprocessing of {this_probe}')

        probe_index = f"probe{i:02d}"

        # Set probe paths
        pp.set_probe_paths(this_probe)
        
        out_path = pp.sorter_path / 'figures'
        os.makedirs(out_path, exist_ok=True)
        
        spikes, clusters, channels = load_neural_data(pp.sorter_path, keep_units='ibl')
        
        u_clust = np.unique(spikes['clusters'])
        u_clust_colors = sns.color_palette('nipy_spectral', n_colors=len(u_clust), as_cmap=False)
        
        di, trains, pulses = load_trig_times(pp.alf_path)
        
        for dich, onoff in di.items():
            assert np.all(onoff['offsets'] - onoff['onsets'])
            fig = plt.figure()
            axs = plot_raster_psth(
                spikes=spikes,
                times=onoff['onsets'],
                pre_ms=300,
                post_ms=800,
                clusters=u_clust,
                cluster_color=u_clust_colors,
                event_color=onoff['color'],
                ttl=f'{dich} Onset',
                fig=fig
            )
            fig.savefig(out_path / '_'.join([dich, 'onsets']))
            
            fig = plt.figure()
            axs = plot_psth(
                spikes=spikes,
                times=onoff['offsets'],
                pre_ms = 300,
                post_ms = 800,
                color=onoff['color'],
                ttl=f'{dich} Offset',
                fig=fig
            )
            fig.savefig(out_path / '_'.join([dich, 'offsets']))
        
        if np.sort(np.unique(pulses['train'])) != np.arange(len(trains['onsets'])):
            print("\nDetected train onsets do not match number of trains")
            print(rec_path)
            
        for tr_i in range(len(trains['onsets'])):
            fig = plt.figure()
            p_idx = pulses['train'] == tr_i
            axs = plot_psth(
                spikes=spikes,
                times=trains['onsets'][tr_i],
                pre_ms = 300,
                post_ms = 800,
                ttl=f'{dich} Offset',
                fig=fig,
                markers=pulses['times'][p_idx],
                marker_kws={
                    'color': 'k'
                },
            )
            i_mean = np.mean(pulses['i_amps'][p_idx])
            i_std = np.std(pulses['i_amps'][p_idx])
            v_mean = np.mean(pulses['v_amps'][p_idx])
            v_std = np.std(pulses['v_amps'][p_idx])
            fig.suptitle(f"VNS $\mu \pm \sigma$\n{i_mean:0.2d}$\pm${i_std:0.2d} mA\n{v_mean:0.2d}$\pm${v_std:0.2d}V")
        
        


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main(Path(r"C:/Users/ixb240017/Box/Neuropixels_Sharing/CATGT/20260218_ABATE125/LC_g3_t0"))
