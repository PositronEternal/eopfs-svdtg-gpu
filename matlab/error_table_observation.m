% generate tables for each particle count set
seq_names = unique({results.name});
particle_counts = unique([results.particle_count]);
score_types = flip(unique({results.score_type}));
filter_modes = flip(unique({results.filter_mode}));

table_path = fullfile(results(1).save_path, 'tables', 'error');
mkdir(table_path);

for particle_count_c = particle_counts
    particle_count = particle_count_c(:);
    particle_count_str = ['pc_' num2str(particle_count)];
    
    clear M M2
    M.Sequence = {};
    
    for seq_name_cell = seq_names
        seq_name = seq_name_cell{:};
        M.Sequence{end+1} = seq_name;

        for filter_mode_c = filter_modes
            for score_type_c = score_types([1 3 2])
                score_type = score_type_c{:};
                filter_mode = filter_mode_c{:};

                errors = results(strcmp({results.name},seq_name));                
                errors = errors([errors.particle_count] == particle_count);
                errors = errors(strcmp({errors.score_type},score_type));
                errors = errors(strcmp({errors.filter_mode},filter_mode));
                
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

                mode = [filter_mode '_' score_type];
                if ~isfield(M, mode)
                    M.(mode) = [];
                end
                M.(mode) = [M.(mode); mse];
            end
        end
    end
    M.Sequence = M.Sequence';
    T = struct2table(M);
    writetable(T, fullfile(table_path, ['mse_table_observation_' particle_count_str '.xlsx']));
    
    % write latex version    
    M2.Sequence = M.Sequence;
    field_names = fieldnames(M)';
    field_names = field_names(2:end);
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
    tex_name = fullfile(table_path, ['mse_table_observation_' particle_count_str '.tex']);
    file_id = fopen(tex_name, 'w');
    [nrows,ncols] = size(latex);
    for row = 1:nrows
        new_text = strrep(latex{row,:}, '_', ' ');
        new_text = strrep(new_text, 'MyTableCaption', ...
        ['MSE Over Observation Methods for \(N=' num2str(particle_count) '\)']);
        new_text = strrep(new_text, 'MyTableLabel', ['mse_observation' num2str(particle_count)]);
        fprintf(file_id, '%s\n', new_text);
    end
    fclose(file_id);
end