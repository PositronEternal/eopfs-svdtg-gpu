track_path = fullfile(results(1).save_path, 'figures', 'track_plots');
mkdir(track_path);

numplots = height(T);
for row_idx = 1:numplots
    seq_name = T{row_idx,1}{:};
    data = T{row_idx,2:end};
    colnames = T(row_idx,2:end).Properties.VariableNames;
    c = categorical(colnames);
    % oh for crying out loud...
    c = reordercats(c, colnames);
    fig = figure;
    bar(c,data);
    set(gca, 'TickLabelInterpreter', 'none')
    
    f = fullfile(track_path, ['track_' seq_name '.eps']);
    print(fig, '-depsc', f);
    close(fig);
end