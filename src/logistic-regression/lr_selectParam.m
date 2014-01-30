function [BEST_AP, BEST_LAMBDA, BEST_ALPHA] =...
    lr_selectParam(classes, Xtrain, ytrain, Xval, yval)
%% init
% clear ; close all; clc;
globals();

%% ============= Part 2: Regularization and Accuracies =============
fprintf('start selecting best params...\n');

% For polynomial
% LAMBDAS = [0.01 0.02 0.025 0.03 0.035 0.04 0.06 0.1 0.3 0.6];
% ALPHAS  = [10 30 50 55 60 65 70 100 150 200 300];

% For no polynomial/normalization
% LAMBDAS = [0.02 0.0225 0.025 0.0275 0.03 0.0325 0.035];% 0.04 0.06 0.1];
% ALPHAS  = [100 125 150 175 200];

% For no polynomial/normalization + BoF features
% LAMBDAS = [0.003 0.004 0.005 0.006 0.007];
% ALPHAS  = [70 75 80];

% For no polynomial/normalization + BoF features + Bndbox
%LAMBDAS = [0.001 0.003 0.006 0.01 0.013 0.016 0.02];
%ALPHAS  = [80 90 100 110 120];

% For no polynomial/normalization + BoF features + Bndbox + SIFT
LAMBDAS = [0.001 0.005 0.01 0.02];
ALPHAS  = [80];

ITER = 2000;  % 2000

BEST_AP = 0;
BEST_LAMBDA = 0;
BEST_ALPHA = 0;

for idx_lambda=1:length(LAMBDAS)
    LAMBDA = LAMBDAS(1,idx_lambda);
    for idx_alpha=1:length(ALPHAS)
        ALPHA = ALPHAS(1,idx_alpha);
        fprintf('LAMBDA: %0.5f, ALPHA: %0.5f...\n', LAMBDA, ALPHA);
        
        filename = fullfile(cache_folder, '/',...
            sprintf('lr_thetas_L%0.5f_A%0.5f_I%d.mat', LAMBDA, ALPHA, ITER))
        if exist(filename, 'file')
            load(filename);
        else
            thetas = [];
            for i=1:size(classes,2)
                fprintf('start %s\n', classes{i});
                theta = zeros(size(Xtrain, 2), 1);
                Jhist = [];
                for j=1:ITER
                    [J, grad] = costFunctionReg(theta, Xtrain,...
                        ytrain(:,i), LAMBDA);
                    theta = theta - ALPHA * grad;
                    Jhist(end+1, :) = [j, J];
                    if(mod(j, 100) == 0)
                        fprintf('cost : %0.4f\n', J);
                    end
                end
                if 0
                    figure;
                    hold on;
                    plot(Jhist(:,1), Jhist(:,2));
                    title(sprintf('LAMBDA: %0.5f, ALPHA: %0.5f', LAMBDA, ALPHA));
                    %pause;
                    hold off;
                    %close(111);
                end
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
            BEST_ALPHA = ALPHA;
        end
    end
end    
fprintf('done\n');

end