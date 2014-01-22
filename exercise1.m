function exercise1
%% SEE Line 47 and Line 64 for exercises.

    rand('seed', 0);
    
    %% Visualization (NO NEED TO MODIFY)
    X = [randn(100,2); randn(100,2)+10];
    for k = 2:5
        figure(k);
        y = kMeans(X, k);
        visualize_result(k, X, y);
    end
    
    X = [randn(100,2); randn(100,2)];
    for k = 2:5
        figure(k+5);
        y = kMeans(X, k);
        visualize_result(k, X, y);
    end
end

function visualize_result(k, X, y)
    %% Visualization (no need to update)
    colors = 'rgbky';
    clf;
    hold on;
    for i = 1:k
        plot(X(y==i,1), X(y==i,2), [colors(i) 'o']);
    end
    hold off;
end

function last_assignment = kMeans(data,k)
    % N is number of examples, d is number of feature dimension
    [N, d] = size(data);
    
    if N <= k,
        error('k has to be larger than number of samples');
    else
        % Initialize cluster centers
        centroid = zeros(k, d);
        for i = 1:k
            centroid(i,:) = data(i,:);
        end

        last_assignment = zeros(N, 1); % initialize as zero vector
        while (1)
            %% Step1: Assign data points to closest cluster center
            
            % Compute distance from each example to all cluster centers
            % and assign each example to its closest cluster
            % Take a look at funcitons [pdist2] and [min]
            % EDIT HERE.........
            
            D = pdist2(data, centroid);
            [~, tmp] = min(D, [], 2);
            
            if tmp == last_assignment
                % if new assignment is the same as the last one, then no
                % more new update
                break;
            else
                % if new assignmetn changed from the last one, update the
                % assignment and keep going
                last_assignment = tmp;
            end
            
            %% Step2: Change cluster center to its updated average
            for i = 1:k
                % Find all examples assigned to i
                % Extract all data assigned to cluster i and put them into
                % [cluster_members]
                % EDIT HERE.......
                cluster_members = data(find(last_assignment == i), :);
                
                % Assign the average of all members in cluster i and assign
                % the average into centroid(i,:)
                centroid(i,:) = mean(cluster_members);
            end
        end
    end
end