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
        pp.extract_sync_pulses()
        pp.extract_stim_pulses()

    # Loop over multiple probes
    raw_probes = spikeglx.get_probes_from_folder(pp.session_path)
    for i, this_probe in enumerate(raw_probes):
        print(f'\nStarting preprocessing of {this_probe}')

        probe_index = f"probe{i:02d}"

        # Set probe paths
        pp.set_probe_paths(this_probe)

        # Decompress raw data if necessary
        print(f'\nDecompressing {this_probe}')
        pp.decompress()

        print(f'\nConverting {this_probe} metadata to kilosort format')
        meta_path_dicts = spikeglx.glob_ephys_files(pp.session_path / 'raw_ephys_data', ext='meta')
        for meta_path_dict in meta_path_dicts:
            if 'ap' in meta_path_dict:
                MetaToCoords(meta_path_dict['ap'], outType=1)

        print(f'\nLoading {this_probe} recording')
        # Load in raw data
        rec = pp.load_raw_binary()

        # Preprocessing
        rec = pp.preprocessing(rec)

        # Spike sorting
        print(f'\nStarting {this_probe} spike sorting at {datetime.now().strftime("%H:%M")}')
        sort = pp.spikesorting(rec)
        if sort is None:
            print('Sorting failed!')
            continue
        print(f'Detected {sort.get_num_units()} units\n')

        # Create sorting analyzer for manual curation in SpikeInterface and save to disk
        pp.neuron_metrics(sort, rec)

        # Export sorting results and LFP metrics
        pp.export_data(rec)

        # Add indication if neurons are good from several sources to the quality metrics
        pp.automatic_curation()

        # Synchronize spike sorting to the nidq clock
        if pp.settings['USE_NIDAQ'] or pp.settings['FORCE_NIDAQ']:
            pp.probe_synchronization()

        # Compress raw data
        pp.compress_raw_data()
        
        with open(pp.alf_path / 'powerpixels_settings.json', 'w') as f:
            json.dump(pp.settings_dict(), f, indent=1)

        print(f'Done! At {datetime.now().strftime("%H:%M")}')

# Delete process_me.flag if all probes are processed
# if np.sum(probe_done) == len(probes):
#     os.remove(os.path.join(root, 'process_me.flag'))


def main(config_path, data_path):
    # Search for process_me.flag
    for root, directory, files in os.walk(data_path):
        if 'process_me.flag' in files:
            try:
                run(config_path, root)
            except Exception as e:
                warning = RuntimeWarning(*e.args)
                warning.with_traceback(e.__traceback__)
                print(warning)
                pass


if __name__ == "__main__":
    data_prefix = Path.home() / "Box/Neuropixels_Sharing/CATGT/"
    data_path = data_prefix #/ "20260218_ABATE125" / "LC_g6_t6"
    if len(sys.argv) < 2:
        config_path = Path(__file__).parent.parent / 'config' / 'settings.json'
    else:
        config_path = Path(sys.argv[1])
        if len(sys.argv) > 2:
            data_path = data_prefix / sys.argv[2]
    main(config_path, data_path)
