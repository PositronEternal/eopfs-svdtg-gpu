seq_names = unique({results.name});
for seq_name_cell = seq_names
    seq_name = seq_name_cell{:};
    
    score_types = unique({results(strcmp({results.name},seq_name)).score_type});
    filter_modes = unique({results(strcmp({results.name},seq_name)).filter_mode});
    
    plot_count = numel(score_types)*numel(filter_modes);
    plot_index = 1;
    figure('Name',seq_name,'NumberTitle','off');
    for filter_mode_cell = flip(filter_modes)
        for score_type_cell = flip(score_types)
            score_type = score_type_cell{:};
            filter_mode = filter_mode_cell{:};
            
            % gather neff values
            neffs = results(strcmp({results.name},seq_name));
            neffs = neffs(strcmp({neffs.score_type},score_type));
            neffs = neffs(strcmp({neffs.filter_mode},filter_mode));
            neffs = [neffs.neff];
            subplot(plot_count,1,plot_index);
            histogram(neffs, 1:5:300);
            title([score_type ' ' filter_mode]);
            plot_index = plot_index + 1;
        end
    end
end