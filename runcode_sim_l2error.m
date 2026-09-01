% Simulation experiment for l2 error OLS vs MLE for Hawkes model
% Four sections:
%   case1_1: BR 0.80, non-capped beta design
%   case1_2: BR 0.80, capped beta design
%   case2_1: BR 0.98, non-capped beta design
%   case2_2: BR 0.98, capped beta design
% K represents the number of kernels

addpath(genpath(pwd));
clear; clc;

rng(42,'twister');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sim = 2; T = 10000;

% User specified grid for bin size
grid_select = logspace(-6.5,2,120).';
grid_val    = grid_select(1:70)';

% User specified folder name where all the results will be saved
% Here the results will be saved in 'results_sim_error'
results_dir = fullfile(pwd,'results_sim_error');

if ~exist(results_dir,'dir')
    mkdir(results_dir);
end

mu = 1;

% True-parameter lower bounds
alphaMin = 2;   % implies min(alpha_k) > 1
betaMin  = 3;   % implies min(beta_k)  > 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select section: specify case1_1, case1_2, case2_1, case2_2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RUN_CASE_ID = 'case1_1';

switch RUN_CASE_ID
    
    case 'case1_1'
        run_case( ...
            Sim,T,grid_val,results_dir, ...
            'BR08',0.80,'case1_1',mu,true, ...
            alphaMin,betaMin);
        
    case 'case1_2'
        run_case( ...
            Sim,T,grid_val,results_dir, ...
            'BR08',0.80,'case1_2',mu,true, ...
            alphaMin,betaMin);
        
    case 'case2_1'
        run_case( ...
            Sim,T,grid_val,results_dir, ...
            'BR098',0.98,'case2_1',mu,true, ...
            alphaMin,betaMin);
        
    case 'case2_2'
        run_case( ...
            Sim,T,grid_val,results_dir, ...
            'BR098',0.98,'case2_2',mu,true, ...
            alphaMin,betaMin);
        
    otherwise
        error('Unknown RUN_CASE_ID: %s',RUN_CASE_ID);
end