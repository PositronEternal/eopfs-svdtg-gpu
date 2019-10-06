% generate tables for each particle count set
particle_counts = unique([results.particle_count]);
update_intervals = unique([results.update_interval]);
update_methods = unique({results.update_method});
historical_lengths = unique([results.historical_length]);

table_path = fullfile(results(1).save_path, 'tables', 'track');
mkdir(table_path);

for particle_count_c = particle_counts
    particle_count = particle_count_c(:);
    particle_count_str = ['pc_' num2str(particle_count)];
    
    clear M M2
    M = {};

    for update_method_cell = update_methods
        for update_interval_cell = update_intervals
            for historical_length_cell = historical_lengths
                update_method = update_method_cell{:};
                update_interval = update_interval_cell(:);
                historical_length = historical_length_cell(:);

                tracks = results([results.particle_count] == particle_count);
                tracks = tracks(strcmp({tracks.update_method},update_method));
                tracks = tracks([tracks.update_interval] == update_interval);
                tracks = tracks([tracks.historical_length] == historical_length);

                % gather track lengths
                tracking = ~[tracks.track_lost];
                tracksum = sum(tracking == true);
                trackavg = tracksum/numel(tracks);

                mode = [update_method '_i' int2str(update_interval) '_h' int2str(historical_length)];
                if ~isfield(M, mode)
                    M.(mode) = [];
                end
                M.(mode) = [M.(mode); trackavg];

            end
        end
    end
    T = struct2table(M);
    writetable(T, fullfile(table_path, ['track_table_updates_composite_' particle_count_str '.xlsx']));
    
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
    tex_name = fullfile(table_path, ['track_table_updates_composite_' particle_count_str '.tex']);
    file_id = fopen(tex_name, 'w');
    [nrows,ncols] = size(latex);
    for row = 1:nrows
        new_text = strrep(latex{row,:}, '_', ' ');
        new_text = strrep(new_text, 'MyTableCaption', ...
        ['Composite Average track length across template update methods for \(N=' num2str(particle_count) '\)']);
        new_text = strrep(new_text, 'MyTableLabel', ['average_track_updates_composite' num2str(particle_count)]);
        fprintf(file_id, '%s\n', new_text);
    end
    fclose(file_id);
end