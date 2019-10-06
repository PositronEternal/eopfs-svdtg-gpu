particle_counts = unique([results.particle_count]);
score_types = flip(unique({results.score_type}));
filter_modes = flip(unique({results.filter_mode}));

table_path = fullfile(results(1).save_path, 'tables', 'neff');
mkdir(table_path);

for particle_count_c = particle_counts
    particle_count = particle_count_c(:);
    particle_count_str = ['pc_' num2str(particle_count)];
    
    clear M M2
    M = {};
    
    % gather histogram values
    for filter_mode_c = filter_modes
        for score_type_c = score_types([1 3 2])
            score_type = score_type_c{:};
            filter_mode = filter_mode_c{:};

            neffs = results([results.particle_count] == particle_count);
            neffs = neffs(strcmp({neffs.score_type},score_type));
            neffs = neffs(strcmp({neffs.filter_mode},filter_mode));
            track_losts = [neffs.track_lost];
            neffs = [neffs.neff];
            neffs = neffs(~track_losts);
            neffs = neffs(~isnan(neffs));

            mode = [filter_mode '_' score_type];
            if ~isfield(M, mode)
                M.(mode) = [];
            end
            med = median(neffs);
            M.(mode) = [M.(mode); median(neffs)];
        end
    end
    T = struct2table(M);
    writetable(T, fullfile(table_path, ['neff_table_composite_' particle_count_str '.xlsx']));
    
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
    tex_name = fullfile(table_path, ['neff_table_composite_' particle_count_str '.tex']);
    file_id = fopen(tex_name, 'w');
    [nrows,ncols] = size(latex);
    for row = 1:nrows
        new_text = strrep(latex{row,:}, '_', ' ');
        new_text = strrep(new_text, 'MyTableCaption', ...
        ['Composite Effective Number of Particles for \(N=' num2str(particle_count) '\)']);
        new_text = strrep(new_text, 'MyTableLabel', ['neff_composite' num2str(particle_count)]);
        fprintf(file_id, '%s\n', new_text);
    end
    fclose(file_id);
end