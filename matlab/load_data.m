% load data from results of batch run
results_file = '/mnt/data/results/results.mat';
load(results_file, 'results');
results = cell2mat(results);

for index = 1:numel(results)
    % discover and store track losses and frame
    frame_count = numel(results(index).frame_number);
    track_lost = true(1,frame_count);
    for frame_index = 1:frame_count
        frame_gt = results(index).ground_truth(:,frame_index);
        frame_est = results(index).estimate(frame_index,:);
        out_of_bounds = check_bounds(frame_gt, frame_est);
        
        if out_of_bounds
            results(index).loss_frame = results(index).frame_number(1,frame_index);
            break
        else
            track_lost(1,frame_index) = false;
        end
        results(index).track_lost = track_lost;
    end
end

function out_of_bounds = check_bounds(gt, x)
    m = x(2);
    n = x(1);
    m_l = gt(1)-gt(3);
    m_h = gt(1)+gt(3);
    n_l = gt(2)-gt(4);
    n_h = gt(2)+gt(4);
    
    m_is_out = m < m_l || m > m_h;
    n_is_out = n < n_l || n > n_h;
    
    out_of_bounds = m_is_out || n_is_out;
end