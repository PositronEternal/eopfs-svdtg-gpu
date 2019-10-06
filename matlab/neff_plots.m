seq_names = unique({results.name});
particle_counts = unique([results.particle_count]);
score_types = flip(unique({results.score_type}));
filter_modes = flip(unique({results.filter_mode}));

fig_path = fullfile(results(1).save_path, 'figures', 'neff_plots');
neff_hists = {};

for seq_name_cell = seq_names
    seq_name = seq_name_cell{:};
    
    neff_hists.(seq_name) = {};
    
    for particle_count_cell = particle_counts
        particle_count = particle_count_cell(:);
        particle_count_str = ['pc_' num2str(particle_count)];        

        % gather histogram values
        for filter_mode_c = filter_modes
            for score_type_c = score_types
                score_type = score_type_c{:};
                filter_mode = filter_mode_c{:};
                
                % gather neff values
                neffs = results(strcmp({results.name},seq_name));
                neffs = neffs([neffs.particle_count] == particle_count);
                neffs = neffs(strcmp({neffs.score_type},score_type));
                neffs = neffs(strcmp({neffs.filter_mode},filter_mode));
                track_losts = [neffs.track_lost];
                neffs = [neffs.neff];
                neffs = neffs(~track_losts);
                edges = 0:5:particle_count;
                neff_hists.(['edges_' particle_count_str]) = edges;
                neff_hist = histcounts(neffs, edges);
                neff_hists.(seq_name).(particle_count_str).(score_type).(filter_mode) = neff_hist;
            end
        end
    
        % plot verticle comparison histograms
        p = fullfile(fig_path, particle_count_str);
        f = fullfile(p, ['neff_' seq_name '_' particle_count_str '.eps']);
        if exist(f, 'file')
            % do nothing
        else
            plot_count = numel(score_types)*numel(filter_modes);
            plot_index = 1;
            fig = figure('Name',[seq_name ' ' particle_count_str(4:end)],'NumberTitle','off');
            for filter_mode_c = filter_modes
                for score_type_c = score_types([1 3 2])
                    score_type = score_type_c{:};
                    filter_mode = filter_mode_c{:};

                    edges = neff_hists.(['edges_' particle_count_str]);
                    hist_vals = neff_hists.(seq_name).(particle_count_str).(score_type).(filter_mode);
                    subplot(plot_count,1,plot_index);
                    bar(edges(2:end), hist_vals);
                    set(gca, 'ytick', []);
                    title([score_type ' ' filter_mode]);
                    plot_index = plot_index + 1;
                end
            end

            mkdir(p);
            print(fig, '-depsc', f);
            %close(fig);
        end
    end
end