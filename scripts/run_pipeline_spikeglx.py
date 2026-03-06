# -*- coding: utf-8 -*-
"""
Written by Guido Meijer

"""
# flake8: noqa E501

from powerpixels import Pipeline

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import spikeglx
import spikeinterface.full as si

from SGLXMetaToCoords import MetaToCoords


def run(config_path, data_path):
    log_f = open(Path(data_path) / 'pipeline.log', 'w')
    log_f.write(f"Config file: {config_path}\nData path: {data_path}")
    pp = Pipeline(config_path, data_path)
    print(f'\nStarting pipeline in {pp.session_path} at {datetime.now().strftime("%H:%M")}\n')

    # Detect data format
    pp.detect_data_format()
    if pp.data_format == 'openephys':
        print('WARNING: You are running the SpikeGLX pipeline on an OpenEphys recording!')
        return

    # Initialize NIDAQ synchronization
    if pp.settings['USE_NIDAQ']:
        pp.set_nidq_paths()
        ok = pp.extract_sync_pulses()
        if ok:
            log_f.write(f"\tSync pulses extracted\n")
        ok = pp.extract_stim_pulses()
        if ok:
            log_f.write(f"\tStim pulses extracted\n")

    # Loop over multiple probes
    raw_probes = spikeglx.get_probes_from_folder(pp.session_path)
    for i, this_probe in enumerate(raw_probes):
        print(f'\nStarting preprocessing of {this_probe}')

        probe_index = f"probe{i:02d}"
        log_f.write(f"\tProbe: {this_probe}\n")

        # Set probe paths
        pp.set_probe_paths(this_probe)

        # Decompress raw data if necessary
        print(f'\nDecompressing {this_probe}')
        ok = pp.decompress()
        if ok:
            log_f.write(f"\t\tDecompressed\n")

        print(f'\nConverting {this_probe} metadata to kilosort format')
        meta_path_dicts = spikeglx.glob_ephys_files(pp.session_path / 'raw_ephys_data', ext='meta')
        for meta_path_dict in meta_path_dicts:
            if 'ap' in meta_path_dict:
                MetaToCoords(meta_path_dict['ap'], outType=1)
                log_f.write(f"\t\tKS metadata generated\n")

        # Synchronize spike sorting to the nidq clock
        if pp.settings['USE_NIDAQ'] or pp.settings['FORCE_NIDAQ']:
            ok = pp.probe_synchronization()
            if ok:
                log_f.write(f"\t\tDecompressed\n")

        print(f'\nLoading {this_probe} recording')
        # Load in raw data
        rec = pp.load_raw_binary()
        if rec is not None:
            log_f.write(f"\t\tData loaded\n")

        # Preprocessing
        rec = pp.preprocessing(rec)
        if rec is not None:
            log_f.write(f"\t\tData preprocessed\n")

        # Spike sorting
        print(f'\nStarting {this_probe} spike sorting at {datetime.now().strftime("%H:%M")}')
        sort = pp.spikesorting(rec)
        if sort is None:
            print('Sorting failed!')
            continue
        else:
            log_f.write("\t\tSorted\n")
        print(f'Detected {sort.get_num_units()} units\n')

        # Create sorting analyzer for manual curation in SpikeInterface and save to disk
        ok = pp.neuron_metrics(sort, rec)
        if ok:
            log_f.write("\t\tMetrics Computed\n")

        # Export sorting results and LFP metrics
        ok = pp.export_data(rec)
        if ok:
            log_f.write("\t\tData exported\n")

        # Add indication if neurons are good from several sources to the quality metrics
        ok = pp.automatic_curation()
        if ok:
            log_f.write("\t\tCuration complete\n")

        # Synchronize spike sorting to the nidq clock
        if pp.settings['USE_NIDAQ'] or pp.settings['FORCE_NIDAQ']:
            ok = pp.spike_synchronization()
            if ok:
                log_f.write("\t\tSpikes synchronized\n")

        # Compress raw data
        ok = pp.compress_raw_data()
        if ok:
            log_f.write("\t\tData compressed\n")
        
        with open(pp.alf_path / 'powerpixels_settings.json', 'w') as f:
            json.dump(pp.settings_dict(), f, indent=1)
        
        log_f.close()

        print(f'Done! At {datetime.now().strftime("%H:%M")}')

# Delete process_me.flag if all probes are processed
# if np.sum(probe_done) == len(probes):
#     os.remove(os.path.join(root, 'process_me.flag'))


def main(config_path, data_path):
    # Search for process_me.flag
    for root, directory, files in os.walk(data_path):
        if 'process_me.flag' in files:
            # run(config_path, root)
            try:
                run(config_path, root)
            except Exception as e:
                warning = RuntimeWarning(*e.args)
                warning.with_traceback(e.__traceback__)
                print(warning)
                pass


if __name__ == "__main__":
    data_prefix = Path.home() / "Box/Neuropixels_Sharing/CATGT/"
    data_path = data_prefix / "20260227_LCP02"# / "LC_g0_t3"
    if len(sys.argv) < 2:
        config_path = Path(__file__).parent.parent / 'config' / 'settings.json'
    else:
        config_path = Path(sys.argv[1])
        if len(sys.argv) > 2:
            data_path = data_prefix / sys.argv[2]
    main(config_path, data_path)
