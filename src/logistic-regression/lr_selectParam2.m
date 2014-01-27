function [BEST_AP, BEST_LAMBDA] = lr_selectParam2(classes,...
    Xtrain, ytrain, Xval, yval)
%% init
% clear ; close all; clc;
globals();

%% ============= Part 2: Regularization and Accuracies =============
fprintf('start selecting best params...\n');
LAMBDAS = [0.01 0.03 0.1 0.3 1.0 3.0 10 30];

BEST_AP = 0;
BEST_LAMBDA = 0;

for idx_lambda=1:length(LAMBDAS)
    LAMBDA = LAMBDAS(1,idx_lambda);
    fprintf('LAMBDA: %0.3f...\n', LAMBDA);
    
    filename = fullfile(cache_folder, '/',...
        sprintf('lr_thetas_L%0.5f.mat', LAMBDA));
    if exist(filename, 'file')
        load(filename);
    else
        thetas = [];
        for i=1:size(classes,2)
            fprintf('start %s\n', classes{i});
            theta = zeros(size(Xtrain, 2), 1);
            
            % Set Options
            options = optimset('GradObj', 'on', 'MaxIter', 400);
            % Optimize
            [theta, J, exit_flag] = ...
                fminunc(@(t)(costFunctionReg(t, Xtrain, ytrain(:,i), LAMBDA)),...
                theta, options);
            thetas(:,end+1) = theta;
            fprintf('done %s\n', classes{i});
        end
        save(filename, 'thetas');
    end
    fprintf('done\n');
    AP = lr_computeAP(classes, thetas, Xval, yval);
    if(AP > BEST_AP)
        BEST_AP = AP;
        BEST_LAMBDA = LAMBDA;
    end
end    
fprintf('done\n');

end