% generate tables for each particle count set
particle_counts = unique([results.particle_count]);
score_types = flip(unique({results.score_type}));
filter_modes = flip(unique({results.filter_mode}));

table_path = fullfile(results(1).save_path, 'tables', 'track');
mkdir(table_path);

for particle_count_c = particle_counts
    particle_count = particle_count_c(:);
    particle_count_str = ['pc_' num2str(particle_count)];
    
    clear M M2
    M = {};
    
    for filter_mode_c = filter_modes
        for score_type_c = score_types([1 3 2])
            score_type = score_type_c{:};
            filter_mode = filter_mode_c{:};
                
            tracks = results([results.particle_count] == particle_count);
            tracks = tracks(strcmp({tracks.score_type},score_type));
            tracks = tracks(strcmp({tracks.filter_mode},filter_mode));

            % gather track lengths
            tracking = ~[tracks.track_lost];
            tracksum = sum(tracking == true);
            trackavg = tracksum/numel(tracks);

            mode = [filter_mode '_' score_type];
            if ~isfield(M, mode)
                M.(mode) = [];
            end
            M.(mode) = [M.(mode); trackavg];
        end
    end
    T = struct2table(M);
    writetable(T, fullfile(table_path, ['track_table_observations_composite_' particle_count_str '.xlsx']));
    
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
    tex_name = fullfile(table_path, ['track_table_observations_composite_' particle_count_str '.tex']);
    file_id = fopen(tex_name, 'w');
    [nrows,ncols] = size(latex);
    for row = 1:nrows
        new_text = strrep(latex{row,:}, '_', ' ');
        new_text = strrep(new_text, 'MyTableCaption', ...
        ['Composite Average track length across observation methods for \(N=' num2str(particle_count) '\)']);
        new_text = strrep(new_text, 'MyTableLabel', ['average_track_observation_composite' num2str(particle_count)]);
        fprintf(file_id, '%s\n', new_text);
    end
    fclose(file_id);
end