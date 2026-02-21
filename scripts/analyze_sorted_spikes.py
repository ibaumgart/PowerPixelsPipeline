import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import windows
from scipy.ndimage import convolve1d as convolve

import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import spikeglx
from powerpixels import Pipeline, load_neural_data

FR_STD_EPS = 1

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
    return onsets, offsets


def load_trig_times(alf_path):
    di_channels = [
        'jaw_tap',
        'paw_tap',
        'tail_tap',
        'vns_train'
    ]
    di_colors = sns.color_palette('tab10', n_colors=len(di_channels), as_cmap=False)
    di = {}
    for dich, diclr in zip(di_channels, di_colors):
        on = np.load(alf_path / (dich+'_onset.times.npy'))
        off = np.load(alf_path / (dich+'_offset.times.npy'))
        di[dich] = dict(zip(('onsets', 'offsets'), filter_onoff(on, off)))
        di[dich]['color'] = diclr
    
    on = np.load(alf_path / 'vns_train_onset.times.npy')
    off = np.load(alf_path / 'vns_train_offset.times.npy')
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
    frs = np.zeros((len(u_clust), len(bins) - 1))
    for ci, c in enumerate(u_clust):
        c_idx = spikes['clusters'] == c
        fr, _ = np.histogram(spikes['times'][c_idx], bins=bins)
        frs[ci] = fr / np.diff(bins)
    return frs


def smooth_fr(hist_fr, dt, sigma=50.0, axis=-1):
    kern = windows.gaussian((round(sigma / dt) * 6) + 1, sigma / dt)
    kern = kern / sum(kern)
    smooth_fr = convolve(hist_fr, kern, axis=axis, mode='reflect')
    return smooth_fr


def z_score_fr(hist_fr, times, axis=-1):
    shape = list(hist_fr.shape)
    tile = [1 for _ in hist_fr.shape]
    slices = [slice(0, dimi) for dimi in hist_fr.shape]
    shape[axis] = 1
    tile[axis] = hist_fr.shape[axis]
    b_idx = times < 0
    slices[axis] = b_idx
    baseline_mean = np.tile(
        np.mean(hist_fr[*slices], axis=axis).reshape(shape),
        tile
    )
    baseline_std = np.tile(
        np.std(hist_fr[*slices], axis=axis).reshape(shape),
        tile
    )
    baseline_std[baseline_std < FR_STD_EPS] = FR_STD_EPS
    z_fr = (hist_fr - baseline_mean) / baseline_std
    return z_fr


def plot_raster_psth(
    spikes,
    event_times,
    clusters,
    pre_ms=1100,
    post_ms=1100,
    bin_sz=50,
    event_color=None,
    cluster_color=None,
    ttl=None,
    fig=None,
    markers=[],
    marker_kws={
        'color': 'k',
        'alpha': 0.6
    },
    line_kws={
        'errorbar': ("se",1.0),
        'legend': False,
        'linewidth': 0.4
    },
    scatter_kws={
        's': 5,#1,
        'alpha': 0.5,
        'legend': False
    }
):
    if fig is None:
        fig, axs = plt.subplots(3,1, sharex=True, sharey=False, figsize=(12,9))
    else:
        axs = fig.add_subplot(nrows=3, n_cols=1, sharex=False, sharey=False)
    
    if cluster_color is None:
        cluster_color = ['k'] * len(clusters)
    
    ts = []
    lns = []
    clsts = []
    clrs = []
    ln = 1
    for c, clr in zip(clusters, cluster_color):
        val_clust = spikes['clusters'] == c
        for t in event_times:
            val_spikes = np.logical_and(
                spikes['times'][val_clust] >= t-(pre_ms/1000),
                spikes['times'][val_clust] < t+(post_ms/1000)
            )
            ts.append(spikes['times'][val_clust][val_spikes] - t)
            lns.append(np.ones(val_spikes.sum(), dtype=int) * ln)
            clsts.append(np.ones(val_spikes.sum(), dtype=int) * c)
            clrs.extend([clr] * val_spikes.sum())
            ln = ln + 1
    
    ts = np.hstack(ts)
    lns = np.hstack(lns)
    clsts = np.hstack(clsts)
    # clrs = np.concat(clrs, axis=0)
    
    rast_df = pd.DataFrame(data=np.array([ts, lns, clsts]).T, columns=["Time (sec)", "Unit", "Unit ID"])
    
    sns.scatterplot(rast_df, x="Time (sec)", y="Unit", hue="Unit ID", hue_order=clusters, palette=cluster_color, ax=axs[0], **scatter_kws)
    
    ticks = np.arange(0, len(clusters)+10, 10)
    axs[0].set_yticks(ticks*len(event_times))
    axs[0].set_yticklabels([str(int(yt)) for yt in ticks])
    
    bins = np.arange(-pre_ms, post_ms + bin_sz, bin_sz) / 1000
    centers = bins[:-1] + np.diff(bins) / 2
    
    frs = np.zeros((len(event_times), len(clusters), len(centers)))
    ts = np.copy(frs)
    cs = np.copy(frs)
    for i, t in enumerate(event_times):
        frs[i] = compute_fr(spikes, clusters, bins + t)
        ts[i] = np.tile(centers.reshape((1, -1)), (len(clusters), 1))
        cs[i] = np.tile(clusters.reshape((-1, 1)), (1, len(centers)))
    frs = smooth_fr(frs, bin_sz)
    fr_df = pd.DataFrame(
        data=np.array([frs.reshape((-1,)),ts.reshape((-1,)),cs.reshape((-1,))]).T,
        columns=['Firing Rate (Hz)', 'Time (sec)', 'Unit']
    )
    sns.lineplot(
        data=fr_df, x='Time (sec)', y='Firing Rate (Hz)', hue='Unit', hue_order=clusters, palette=cluster_color, ax=axs[1], **line_kws
    )
    
    norm_frs = z_score_fr(frs, centers)
    norm_fr_df = pd.DataFrame(
        data=np.array([norm_frs.reshape((-1,)),ts.reshape((-1,)),cs.reshape((-1,))]).T,
        columns=['Firing Rate (Z)', 'Time (sec)', 'Unit']
    )
    sns.lineplot(
        data=norm_fr_df, x='Time (sec)', y='Firing Rate (Z)', hue_order=clusters, palette=cluster_color, hue='Unit', alpha=0.5, ax=axs[2], **line_kws
    )
    axs[2].axhline(0, color='black', linestyle=':')
    axs[2].plot(np.mean(ts, axis=(0,1)), np.mean(norm_frs, axis=(0,1)), color='black')
    
    for ax in axs.flat:
        for marker in markers:
            ax.axvline(marker, **marker_kws)
            
    axs[2].set_ylim([-10, 10])
    axs[0].set_xlim([-pre_ms / 1000, post_ms / 1000])
    plt.suptitle(ttl)

    return axs

def plot_raster_session(
    spikes,
    event_dict,
    clusters,
    bin_sz=500,
    cluster_color=None,
    ttl=None,
    fig=None,
    markers=[],
    marker_kws={
        'color': 'k',
        'alpha': 0.4
    },
    line_kws={
        'errorbar': ("se",1.0),
        'legend': False,
        'linewidth': 0.4
    },
    scatter_kws={
        's': 5,#1,
        'alpha': 0.5,
        'legend': False
    }
):
    if fig is None:
        fig, axs = plt.subplots(3,1, sharex=True, sharey=False, figsize=(12,9))
    else:
        axs = fig.add_subplot(nrows=3, n_cols=1, sharex=False, sharey=False)
    
    if cluster_color is None:
        cluster_color = ['k'] * len(clusters)
    
    ts = []
    lns = []
    clsts = []
    clrs = []
    ln = 1
    for c, clr in zip(clusters, cluster_color):
        val_spikes = spikes['clusters'] == c
        ts.append(spikes['times'][val_spikes])
        lns.append(np.ones(val_spikes.sum(), dtype=int) * ln)
        clsts.append(np.ones(val_spikes.sum(), dtype=int) * c)
        ln = ln + 1
    
    ts = np.hstack(ts)
    lns = np.hstack(lns)
    clsts = np.hstack(clsts)
    # clrs = np.concat(clrs, axis=0)
    
    rast_df = pd.DataFrame(data=np.array([ts, lns, clsts]).T, columns=["Time (sec)", "Unit", "Unit ID"])
    
    sns.scatterplot(rast_df, x="Time (sec)", y="Unit", hue="Unit ID", hue_order=clusters, palette=cluster_color, ax=axs[0], **scatter_kws)
    
    ticks = np.arange(0, len(clusters)+10, 10)
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels([str(int(yt)) for yt in ticks])
    
    bins = np.arange(np.min(spikes['times']), np.max(spikes['times']) + bin_sz / 1000, bin_sz / 1000)
    centers = bins[:-1] + np.diff(bins) / 2
    
    frs = np.zeros((1, len(clusters), len(centers)))
    ts = np.copy(frs)
    cs = np.copy(frs)
    frs[0] = compute_fr(spikes, clusters, bins)
    ts[0] = np.tile(centers.reshape((1, -1)), (len(clusters), 1))
    cs[0] = np.tile(clusters.reshape((-1, 1)), (1, len(centers)))
    frs = smooth_fr(frs, bin_sz)
    fr_df = pd.DataFrame(
        data=np.array([frs.reshape((-1,)),ts.reshape((-1,)),cs.reshape((-1,))]).T,
        columns=['Firing Rate (Hz)', 'Time (sec)', 'Unit']
    )
    sns.lineplot(
        data=fr_df, x='Time (sec)', y='Firing Rate (Hz)', hue='Unit', hue_order=clusters, palette=cluster_color, ax=axs[1], **line_kws
    )
    
    norm_frs = z_score_fr(frs, -1*centers)
    norm_fr_df = pd.DataFrame(
        data=np.array([norm_frs.reshape((-1,)),ts.reshape((-1,)),cs.reshape((-1,))]).T,
        columns=['Firing Rate (Z)', 'Time (sec)', 'Unit']
    )
    sns.lineplot(
        data=norm_fr_df, x='Time (sec)', y='Firing Rate (Z)', hue_order=clusters, palette=cluster_color, hue='Unit', ax=axs[2], alpha=0.5, **line_kws
    )
    axs[2].axhline(0, color='black', linestyle=':')
    axs[2].plot(np.mean(ts, axis=(0,1)), np.mean(norm_frs, axis=(0,1)), color='black')
    
    for ax in axs.flat:
        labs = []
        mpl_objs = []
        for ch, ch_dict in event_dict.items():
            for i, (on, off) in enumerate(zip(ch_dict['onsets'],ch_dict['offsets'])):
                if i == 0:
                    mpl_objs.append(ax.axvspan(on, off, color=ch_dict['color'], alpha=0.1))
                    labs.append(ch)
                else:
                    ax.axvspan(on, off, color=ch_dict['color'], alpha=0.1)

    axs[2].legend(handles=mpl_objs, labels=labs)
    axs[2].set_ylim([-10, 10])
    plt.suptitle(ttl)

    return axs

def run(rec_path: Path):
    
    print(f'\nPlotting {rec_path}\n')
        
    pp = Pipeline(this_path.parent / 'config' / 'dummy_settings.json', data_path=rec_path)
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
        
        axs = plot_raster_session(
            spikes,
            di,
            u_clust,
            bin_sz=500,
            cluster_color=u_clust_colors,
            ttl=rec_path.name+f' {this_probe}',
            fig=None,
        )
        axs[0].get_figure().savefig(out_path / 'session_raster')
        # plt.show()
        plt.close(axs[0].get_figure())
        
        for dich, onoff in di.items():
            print(dich+' onsets')
            if len(onoff['onsets']) == 0:
                continue
            assert np.all(onoff['offsets'] - onoff['onsets'])
            fig = None#plt.figure()
            axs = plot_raster_psth(
                spikes=spikes,
                event_times=onoff['onsets'],
                clusters=u_clust,
                event_color=onoff['color'],
                cluster_color=u_clust_colors,
                ttl=f'{dich} Onset'.replace('_', ' ').capitalize() + ' N=' + str(len(onoff['offsets'])),
                fig=fig
            )
            axs[0].get_figure().savefig(out_path / '_'.join([dich, 'onsets']))
            # plt.show()
            plt.close(axs[0].get_figure())
            
            print(dich+' offsets')
            fig = None#plt.figure()
            axs = plot_raster_psth(
                spikes=spikes,
                event_times=onoff['offsets'],
                clusters=u_clust,
                event_color=onoff['color'],
                cluster_color=u_clust_colors,
                ttl=f'{dich} Offset'.replace('_', ' ').capitalize() + ' N=' + str(len(onoff['offsets'])),
                fig=fig
            )
            axs[0].get_figure().savefig(out_path / '_'.join([dich, 'offsets']))
            # plt.show()
            plt.close(axs[0].get_figure())
        
        if len(np.unique(pulses['train'])) != len(trains['onsets']):
            print("\nDetected train onsets do not match number of trains")
            print(rec_path)
            
        for tr_i in range(len(trains['onsets'])):
            fig = None#plt.figure()
            p_idx = pulses['train'] == tr_i
            i_mean = np.mean(pulses['i_amps'][p_idx])
            i_std = np.std(pulses['i_amps'][p_idx])
            v_mean = np.mean(pulses['v_amps'][p_idx])
            v_std = np.std(pulses['v_amps'][p_idx])
            freq = 1 / np.mean(np.diff(pulses['times'][p_idx]))
            axs = plot_raster_psth(
                spikes=spikes,
                event_times=[trains['onsets'][tr_i]],
                clusters=u_clust,
                ttl=f"VNS {freq:0.1f} Hz $\mu \pm \sigma$\n{i_mean:0.2f}$\pm${i_std:0.2f} mA {v_mean:0.2f}$\pm${v_std:0.2f}V",
                event_color=trains['color'],
                cluster_color=u_clust_colors,
                fig=fig,
                markers=pulses['times'][p_idx] - trains['onsets'][tr_i]
            )
            axs[0].get_figure().savefig(out_path / '_'.join(['vns_train', str(tr_i), 'onsets']))
            # plt.show()
            plt.close(axs[0].get_figure())
        
        
def main(data_path):
    # Search for process_me.flag
    for root, directory, files in os.walk(data_path):
        if 'process_me.flag' in files:
            try:
                run(Path(root))
            except:
                pass


if __name__ == "__main__":
    default_path = Path(r"C:/Users/ixb240017/Box/Neuropixels_Sharing/CATGT/20260218_ABATE125")
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main(default_path)
