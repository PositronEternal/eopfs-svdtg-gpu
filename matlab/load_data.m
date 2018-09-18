% load data from results of batch run
results_file = '/mnt/data/results/results.mat';
load(results_file, 'results');
results = cell2mat(results);