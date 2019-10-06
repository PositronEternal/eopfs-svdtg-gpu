% generate tables for each particle count set
particle_counts = unique([results.particle_count]);
score_types = flip(unique({results.score_type}));
filter_modes = flip(unique({results.filter_mode}));
update_intervals = unique([results.update_interval]);
update_methods = unique({results.update_method});
historical_lengths = unique([results.historical_length]);

table_path = fullfile(results(1).save_path, 'tables', 'error');
mkdir(table_path);

for particle_count_c = particle_counts
    particle_count = particle_count_c(:);
    particle_count_str = ['pc_' num2str(particle_count)];
    
    clear M M2
    M={};

    for update_method_cell = update_methods
        for update_interval_cell = update_intervals
            for historical_length_cell = historical_lengths
                update_method = update_method_cell{:};
                update_interval = update_interval_cell(:);
                historical_length = historical_length_cell(:);

                errors = results([results.particle_count] == particle_count);
                errors = errors(strcmp({errors.update_method},update_method));
                errors = errors([errors.update_interval] == update_interval);
                errors = errors([errors.historical_length] == historical_length);
                % generate error squared magnitude
                for runs_idx = 1:numel(errors)
                    runs_error = errors(runs_idx).error;
                    errors(runs_idx).error = (runs_error(:,1).^2 + runs_error(:,2).^2)';
                end
                track_losts = [errors.track_lost];
                errors = [errors.error];
                errors = errors(~track_losts);
                errors = errors(~isnan(errors));
                mse = sum(errors)/numel(errors);

                mode = [update_method '_i' int2str(update_interval) '_h' int2str(historical_length)];
                if ~isfield(M, mode)
                    M.(mode) = [];
                end
                M.(mode) = [M.(mode); mse];

            end
        end
    end
    T = struct2table(M);
    writetable(T, fullfile(table_path, ['mse_table_update_composite_' particle_count_str '.xlsx']));
    
    % write latex version    
    field_names = fieldnames(M)';
    for field_name_c = field_names
        field_name = field_name_c{:};
        M2.(field_name) = {};
        for blah = 1:numel(M.(field_name))
            blah_txt = num2str(M.(field_name)(blah), '%.1f');
            M2.(field_name)(blah,1) = {blah_txt};
        end
    end
    input.data = struct2table(M2);
    latex = latexTable(input);
    tex_name = fullfile(table_path, ['mse_table_update_composite_' particle_count_str '.tex']);
    file_id = fopen(tex_name, 'w');
    [nrows,ncols] = size(latex);
    for row = 1:nrows
        new_text = strrep(latex{row,:}, '_', ' ');
        new_text = strrep(new_text, 'MyTableCaption', ...
        ['Composite MSE Over Template Update Strategies for \(N=' num2str(particle_count) '\)']);
        new_text = strrep(new_text, 'MyTableLabel', ['mse_updates_composite' num2str(particle_count)]);
        fprintf(file_id, '%s\n', new_text);
    end
    fclose(file_id);
end