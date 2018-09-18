from os import path
import json
import h5py
import scipy.io as scio
import numpy as np

batch_config = '/mnt/data/phd/sirlib/batch_config.json'
with open(batch_config, 'r') as bo_file:
    batch_options = json.load(bo_file)

runs = [
    {
        'name': sequence['name'],
        'start_frame': sequence['start_frame'],
        'end_frame': sequence['end_frame'],
        'particle_count': particle_count,
        'score_type': score_type,
        'filter_mode': filter_mode,
        'update_interval': update_interval,
        'update_method': update_method,
        'historical_length': historical_length,
        'run': run,
        'root_path': batch_options['root_path'],
        'save_path': batch_options['save_path']
    }
    for sequence in batch_options['sequences']
    for particle_count in batch_options['particle_counts']
    for score_type in batch_options['score_types']
    for filter_mode in batch_options['filter_modes']
    for update_interval in batch_options['update_intervals']
    for update_method in batch_options['update_methods']
    for historical_length in batch_options['historical_lengths']
    for run in range(batch_options['number_runs'])
]

results = []

for job in runs:

    # load ground-truth
    seq_name = job['name']
    root_path = job['root_path']
    seq_path = path.join(root_path, seq_name)
    gt_path = path.join(seq_path, 'video_params_' + seq_name + '.mat')
    with h5py.File(gt_path) as gt_file:
        g_t = np.array(gt_file.get('gt/save_gt'))
        n_levels = int(gt_file.get('numLevels')[0][0])
        n_orien = int(gt_file.get('numOrien')[0][0])
        height = int(gt_file.get('video_height')[0][0])
        width = int(gt_file.get('video_width')[0][0])
        length = int(gt_file.get('video_length')[0][0])
    n_channels = n_levels * n_orien

    # generate actual end frame and replace if necessary
    end_frame = job['end_frame']
    if end_frame < 0 or end_frame > length:
        job['end_frame'] = length-1

    # load result from json
    result_path = path.join(
        job['save_path'],
        job['name'],
        (str(job['start_frame']) +
            '_' + str(job['end_frame'])),
        'pc_' + str(job['particle_count']),
        job['score_type'],
        job['filter_mode'],
        'ui_' + str(job['update_interval']),
        job['update_method'],
        'hl_' + str(job['historical_length']),
        'results_' + str(job['run']) + '.json'
        )

    with open(result_path, 'r') as r_file:
        job_result = json.load(r_file)

    # save result into memory structure
    results.append(job_result)

scio.savemat(
    path.join(batch_options['save_path'], 'results.mat'),
    {'results': results})
