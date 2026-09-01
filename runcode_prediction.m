% The following code will create txt files and mat files:

% TXT FILE:
% For each trading day and each prediction horizon (h={1,5,10} minutes), 
% the txt file will store one row containing the file name, the horizon, 
% the total number of events that day, and the number of forecasts. 
% For MLE: the txt file will report the average out-of-sample MSE, RMSE,
% and MAE (conditional Hawkes forecast), stationary forecasts (not reported
% in the paper)
% It also reports the average in-sample and out-of-sample negative log-likelihoods and AIC values.
% For OLS, the txt file will report the same quantities as OLS but for each 
% of the 96 values of c: for every c, it contains the OLS MSE, RMSE, MAE, stationary-forecast MSE/RMSE/MAE  (not reported
% in the paper), in-sample OLS loss, in-sample NLL and AIC, and out-of-sample NLL and AIC. 
% The values written to the TXT file are the averages over all forecast origins of that day and horizon.


% MAT FILE:
% For each trading day and forecast horizon, the results structure stores 
% the same aggregate statistics as the TXT file, and it will also retain the 
% underlying forecast-origin-level quantities. It will save the individual 
% squared and absolute prediction errors for MLE and OLS, both for conditional 
% and stationary forecasts; the individual in-sample and out-of-sample NLL/AIC values; 
% the MLE and OLS predicted counts; the realized future counts; 
% and the actual bin widths used at every prediction origin and every c
% It will save the estimated parameters at every prediction origin. 
% For MLE, these are mu_MLE_store, alpha_MLE_store, beta_MLE_store, and the branching ratio eta_MLE_store. 
% For OLS, these are mu_OLS_store, alpha_OLS_store, and eta_OLS_store for all 96 grid values. 
% Since the OLS estimation uses the MLE decay coefficients, there is no separate
% beta_OLS_store: at each prediction origin, the OLS estimator is computed conditional on the corresponding MLE 
    
%%
addpath(genpath(pwd));
clear; clc;

rng(42,'twister');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code upload the data through their csv files, located in the folder 'Data'
% All the data are in csv format, and each csv file represents 1 trading
% day. All csv files are included in the subfolder 'Data/ASSET' 
dataFolder  = fullfile(pwd, 'Data', 'ASSET');
filePattern = 'taqASSET-*.csv';
assetName   = 'AssetName';

% Number of kernels: 2,3,4,5
n_list = 2:5;

sessionStart = duration(9,30,0);
sessionEnd   = duration(15,30,0);

Tfull = seconds(sessionEnd - sessionStart);

rollWindowSec = 2 * 60 * 60;
predStartSec  = seconds(duration(13,0,0) - sessionStart);
predEndSec    = Tfull;

L_list_min = [1 5 10];
L_list_sec = 60 * L_list_min;

grid_val = logspace(-6.5,2,96).';
Mgrid = numel(grid_val);

files = dir(fullfile(dataFolder, filePattern));
if isempty(files)
    error('No CSV files found in folder: %s', dataFolder);
end

[~, idxSort] = sort({files.name});
files = files(idxSort);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options for optimizer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'Display', 'off', ...
    'MaxIterations', 50000, ...
    'MaxFunctionEvaluations', 2e5, ...
    'SpecifyObjectiveGradient', false);

% Loop over the number of kernels
for n = n_list
    
    nParam = 2*n + 1;

    outFile = fullfile(dataFolder, ...
        sprintf('hawkes_prediction_%s_n%d_rolling2h_from1300_allbins.txt', assetName, n));
    
    matFile = fullfile(dataFolder, ...
        sprintf('hawkes_prediction_%s_n%d_rolling2h_from1300_allbins.mat', assetName, n));
    
    fid = fopen(outFile, 'w');
    if fid == -1
        error('Cannot open output file: %s', outFile);
    end
    
    cleanupObj = onCleanup(@() safe_close_files(fid));
    
    fileW = 28;
    intW  = 12;
    numW  = 18;
    
    fprintf(fid, '%-*s', fileW, 'file');
    fprintf(fid, '%*s', intW, 'L_min');
    fprintf(fid, '%*s', intW, 'N');
    fprintf(fid, '%*s', intW, 'nForecast');
    
    fprintf(fid, '%*s', numW, 'MSE_MLE');
    fprintf(fid, '%*s', numW, 'RMSE_MLE');
    fprintf(fid, '%*s', numW, 'MAE_MLE');
    
    fprintf(fid, '%*s', numW, 'MSE_MLE_STAT');
    fprintf(fid, '%*s', numW, 'RMSE_MLE_STAT');
    fprintf(fid, '%*s', numW, 'MAE_MLE_STAT');
    
    fprintf(fid, '%*s', numW, 'NLL_MLE_IS');
    fprintf(fid, '%*s', numW, 'AIC_MLE_IS');
    fprintf(fid, '%*s', numW, 'NLL_MLE_OOS');
    fprintf(fid, '%*s', numW, 'AIC_MLE_OOS');
    
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('MSE_OLS_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('RMSE_OLS_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('MAE_OLS_c%d', m));
    end
    
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('MSE_OLS_STAT_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('RMSE_OLS_STAT_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('MAE_OLS_STAT_c%d', m));
    end
    
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('OLSLOSS_IS_c%d', m));
    end
    
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('NLL_OLS_IS_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('AIC_OLS_IS_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('NLL_OLS_OOS_c%d', m));
    end
    for m = 1:Mgrid
        fprintf(fid, '%*s', numW, sprintf('AIC_OLS_OOS_c%d', m));
    end
    
    fprintf(fid, '\n');
    
    totalWidth = fileW + 3*intW + numW*(10 + 11*Mgrid);
    fprintf(fid, '%s\n', repmat('-', 1, totalWidth));
    drawnow;
    
    results = struct();
    rr = 0;
    
    % Loop over all the trading days
    for ii = 1:numel(files)
        
        fileName = files(ii).name;
        filePath = fullfile(files(ii).folder, fileName);
        
        fprintf('\nn=%d | Processing %d/%d: %s\n', ...
            n, ii, numel(files), fileName);
        
        try
            tbl = readtable(filePath);
            
            if ~ismember('ts', tbl.Properties.VariableNames)
                error('Column "ts" not found in file %s.', fileName);
            end
            
            ts = tbl.ts;
            
            if ~isduration(ts)
                error('Column "ts" is expected to be a duration variable.');
            end
            
            ts = ts(ts >= sessionStart & ts <= sessionEnd);
            
            eventsTimes = seconds(ts - sessionStart);
            eventsTimes = clean_event_times(eventsTimes);
            N = numel(eventsTimes);
            
            if N == 0
                error('No valid events in 09:30--15:30 window.');
            end
            
            baseStepSec = min(L_list_sec);
            predTimesMaster = predStartSec:baseStepSec:(predEndSec - baseStepSec);
            nMaster = numel(predTimesMaster);
            
            fitTimesCell = cell(nMaster,1);
            
            mu_MLE_master    = NaN(nMaster,1);
            alpha_MLE_master = NaN(n,nMaster);
            beta_MLE_master  = NaN(n,nMaster);
            eta_MLE_master   = NaN(nMaster,1);
            
            NLL_MLE_IS_master = NaN(nMaster,1);
            AIC_MLE_IS_master = NaN(nMaster,1);
            
            mu_OLS_master    = NaN(nMaster,Mgrid);
            alpha_OLS_master = NaN(n,Mgrid,nMaster);
            eta_OLS_master   = NaN(nMaster,Mgrid);
            
            OLSLOSS_IS_master = NaN(nMaster,Mgrid);
            NLL_OLS_IS_master = NaN(nMaster,Mgrid);
            AIC_OLS_IS_master = NaN(nMaster,Mgrid);
            
            dtGrid_master = NaN(nMaster,Mgrid);
            status_master = false(nMaster,1);
            
            % Loop over all the training/oos samples (Parallel recommended)
            parfor hh = 1:nMaster
                
                try
                    rng(42 + 100000*n + 1000*ii + hh, 'twister');
                    
                    t0 = predTimesMaster(hh);
                    
                    fitStart = t0 - rollWindowSec;
                    fitEnd   = t0;
                    
                    if fitStart < 0
                        continue;
                    end
                    
                    fitTimesRaw = eventsTimes(eventsTimes >= fitStart & eventsTimes < fitEnd);
                    fitTimes = fitTimesRaw - fitStart;
                    fitTimes = clean_event_times(fitTimes);
                    
                    Tfit = rollWindowSec;
                    Nfit = numel(fitTimes);
                    
                    if Nfit == 0
                        continue;
                    end
                    
                    fitMarks = ones(size(fitTimes));
                    fitTimesCell{hh} = fitTimes;
                    
                    mu0    = max(Nfit / Tfit * 0.2, 1e-3);
                    alpha0 = (-2 + 4*rand(n,1)) + 20*ones(n,1);
                    beta0  = (-2 + 4*rand(n,1)) + 60*ones(n,1);
                    theta_init = [mu0; alpha0; beta0];
                    
                    obj = @(x) HawkesMLE_full(x, fitTimes, fitMarks, Tfit, n);
                    
                    [theta_hat, nll_mle_is] = fmincon( ...
                        obj, theta_init, [], [], [], [], [], [], ...
                        @(x) constr_hawkes_full(x, n), options);
                    
                    mu_mle    = theta_hat(1);
                    alpha_mle = theta_hat(2:1+n);
                    beta_mle  = theta_hat(2+n:end);
                    
                    [beta_mle, ord] = sort(beta_mle);
                    alpha_mle = alpha_mle(ord);
                    
                    eta_mle = sum(alpha_mle ./ beta_mle);
                    
                    mu_MLE_master(hh) = mu_mle;
                    alpha_MLE_master(:,hh) = alpha_mle;
                    beta_MLE_master(:,hh) = beta_mle;
                    eta_MLE_master(hh) = eta_mle;
                    
                    NLL_MLE_IS_master(hh) = nll_mle_is;
                    AIC_MLE_IS_master(hh) = 2*nParam + 2*nll_mle_is;
                    
                    dtGrid = grid_val ./ max(beta_mle);
                    dtGrid_master(hh,:) = dtGrid(:).';
                    
                    mu_ols_all    = NaN(Mgrid,1);
                    alpha_ols_all = NaN(n,Mgrid);
                    eta_ols_all   = NaN(Mgrid,1);
                    
                    ols_loss_tmp   = NaN(Mgrid,1);
                    nll_ols_is_tmp = NaN(Mgrid,1);
                    aic_ols_is_tmp = NaN(Mgrid,1);
                    
                    % Loop over the bin size for OLS estimation
                    for m = 1:Mgrid
                        
                        dt = dtGrid(m);
                        
                        variable = bin_events(fitTimes, Tfit, dt);
                        
                        est_ols = hawkes_OLS(variable.nzIdx, variable.nzVal, ...
                            variable.T, dt, beta_mle);
                        
                        mu_tmp = est_ols.theta(1);
                        alpha_tmp = est_ols.theta(2:end);
                        
                        mu_ols_all(m) = mu_tmp;
                        alpha_ols_all(:,m) = alpha_tmp;
                        eta_ols_all(m) = sum(alpha_tmp ./ beta_mle);
                        
                        ols_loss_tmp(m) = hawkes_OLS_insample_loss( ...
                            variable.nzIdx, variable.nzVal, variable.T, ...
                            dt, beta_mle, est_ols.theta);
                        
                        theta_ols_tmp = [mu_tmp; alpha_tmp; beta_mle];
                        
                        nll_is_tmp = HawkesMLE_full(theta_ols_tmp, ...
                            fitTimes, fitMarks, Tfit, n);
                        
                        aic_is_tmp = 2*nParam + 2*nll_is_tmp;
                        
                        nll_ols_is_tmp(m) = nll_is_tmp;
                        aic_ols_is_tmp(m) = aic_is_tmp;
                    end
                    
                    mu_OLS_master(hh,:) = mu_ols_all(:).';
                    alpha_OLS_master(:,:,hh) = alpha_ols_all;
                    eta_OLS_master(hh,:) = eta_ols_all(:).';
                    
                    OLSLOSS_IS_master(hh,:) = ols_loss_tmp(:).';
                    NLL_OLS_IS_master(hh,:) = nll_ols_is_tmp(:).';
                    AIC_OLS_IS_master(hh,:) = aic_ols_is_tmp(:).';
                    
                    status_master(hh) = true;
                    
                catch
                    status_master(hh) = false;
                end
            end
            
            for ll = 1:numel(L_list_sec)
                
                Lsec = L_list_sec(ll);
                Lmin = L_list_min(ll);
                
                predTimesStart = predStartSec:Lsec:(predEndSec - Lsec);
                nForecast = numel(predTimesStart);
                
                [~, masterIdx] = ismember(predTimesStart, predTimesMaster);
                
                SE_MLE_all      = NaN(nForecast,1);
                AE_MLE_all      = NaN(nForecast,1);
                SE_MLE_STAT_all = NaN(nForecast,1);
                AE_MLE_STAT_all = NaN(nForecast,1);
                
                SE_OLS_all      = NaN(nForecast,Mgrid);
                AE_OLS_all      = NaN(nForecast,Mgrid);
                SE_OLS_STAT_all = NaN(nForecast,Mgrid);
                AE_OLS_STAT_all = NaN(nForecast,Mgrid);
                
                NLL_MLE_IS_all  = NaN(nForecast,1);
                AIC_MLE_IS_all  = NaN(nForecast,1);
                NLL_MLE_OOS_all = NaN(nForecast,1);
                AIC_MLE_OOS_all = NaN(nForecast,1);
                
                OLSLOSS_IS_all  = NaN(nForecast,Mgrid);
                NLL_OLS_IS_all  = NaN(nForecast,Mgrid);
                AIC_OLS_IS_all  = NaN(nForecast,Mgrid);
                NLL_OLS_OOS_all = NaN(nForecast,Mgrid);
                AIC_OLS_OOS_all = NaN(nForecast,Mgrid);
                
                pred_MLE_all      = NaN(nForecast,1);
                pred_MLE_STAT_all = NaN(nForecast,1);
                pred_OLS_all      = NaN(nForecast,Mgrid);
                pred_OLS_STAT_all = NaN(nForecast,Mgrid);
                
                realized_all = NaN(nForecast,1);
                dtGrid_all   = NaN(nForecast,Mgrid);
                
                mu_MLE_store    = NaN(nForecast,1);
                alpha_MLE_store = NaN(n,nForecast);
                beta_MLE_store  = NaN(n,nForecast);
                eta_MLE_store   = NaN(nForecast,1);
                
                mu_OLS_store    = NaN(nForecast,Mgrid);
                alpha_OLS_store = NaN(n,Mgrid,nForecast);
                eta_OLS_store   = NaN(nForecast,Mgrid);
                
                for jj = 1:nForecast
                    
                    hh = masterIdx(jj);
                    if hh <= 0 || ~status_master(hh)
                        continue;
                    end
                    
                    t0 = predTimesStart(jj);
                    t1 = t0 + Lsec;
                    
                    fitStart = t0 - rollWindowSec;
                    fitTimes = fitTimesCell{hh};
                    
                    if isempty(fitTimes)
                        continue;
                    end
                    
                    Tfit = rollWindowSec;
                    fitMarks = ones(size(fitTimes));
                    
                    realizedCount = sum(eventsTimes >= t0 & eventsTimes < t1);
                    realized_all(jj) = realizedCount;
                    
                    predStartLocal = Tfit;
                    predEndLocal   = Tfit + Lsec;
                    
                    testTimesLocal = eventsTimes(eventsTimes >= t0 & eventsTimes < t1) - fitStart;
                    testTimesLocal = clean_event_times(testTimesLocal);
                    
                    mu_mle    = mu_MLE_master(hh);
                    alpha_mle = alpha_MLE_master(:,hh);
                    beta_mle  = beta_MLE_master(:,hh);
                    
                    theta_mle = [mu_mle; alpha_mle; beta_mle];
                    
                    mu_MLE_store(jj) = mu_mle;
                    alpha_MLE_store(:,jj) = alpha_mle;
                    beta_MLE_store(:,jj) = beta_mle;
                    eta_MLE_store(jj) = eta_MLE_master(hh);
                    
                    NLL_MLE_IS_all(jj) = NLL_MLE_IS_master(hh);
                    AIC_MLE_IS_all(jj) = AIC_MLE_IS_master(hh);
                    
                    nll_mle_oos = HawkesNLL_oos_interval(theta_mle, ...
                        fitTimes, testTimesLocal, predStartLocal, predEndLocal, n);
                    aic_mle_oos = 2*nParam + 2*nll_mle_oos;
                    
                    NLL_MLE_OOS_all(jj) = nll_mle_oos;
                    AIC_MLE_OOS_all(jj) = aic_mle_oos;
                    
                    predCount_mle = hawkes_predict_count_interval( ...
                        fitTimes, predStartLocal, predEndLocal, ...
                        mu_mle, alpha_mle, beta_mle);
                    
                    pred_MLE_all(jj) = predCount_mle;
                    
                    predCount_mle_stat = hawkes_predict_count_stationary( ...
                        Lsec, mu_mle, alpha_mle, beta_mle);
                    
                    pred_MLE_STAT_all(jj) = predCount_mle_stat;
                    
                    SE_MLE_all(jj)      = (realizedCount - predCount_mle)^2;
                    AE_MLE_all(jj)      = abs(realizedCount - predCount_mle);
                    
                    SE_MLE_STAT_all(jj) = (realizedCount - predCount_mle_stat)^2;
                    AE_MLE_STAT_all(jj) = abs(realizedCount - predCount_mle_stat);
                    
                    dtGrid_all(jj,:) = dtGrid_master(hh,:);
                    
                    mu_OLS_store(jj,:) = mu_OLS_master(hh,:);
                    alpha_OLS_store(:,:,jj) = alpha_OLS_master(:,:,hh);
                    eta_OLS_store(jj,:) = eta_OLS_master(hh,:);
                    
                    OLSLOSS_IS_all(jj,:) = OLSLOSS_IS_master(hh,:);
                    NLL_OLS_IS_all(jj,:) = NLL_OLS_IS_master(hh,:);
                    AIC_OLS_IS_all(jj,:) = AIC_OLS_IS_master(hh,:);
                    
                    for m = 1:Mgrid
                        
                        mu_tmp = mu_OLS_master(hh,m);
                        alpha_tmp = alpha_OLS_master(:,m,hh);
                        
                        if ~isfinite(mu_tmp) || any(~isfinite(alpha_tmp))
                            continue;
                        end
                        
                        theta_ols_tmp = [mu_tmp; alpha_tmp; beta_mle];
                        
                        pred_tmp = hawkes_predict_count_interval( ...
                            fitTimes, predStartLocal, predEndLocal, ...
                            mu_tmp, alpha_tmp, beta_mle);
                        
                        pred_OLS_all(jj,m) = pred_tmp;
                        
                        pred_stat_tmp = hawkes_predict_count_stationary( ...
                            Lsec, mu_tmp, alpha_tmp, beta_mle);
                        
                        pred_OLS_STAT_all(jj,m) = pred_stat_tmp;
                        
                        nll_oos_tmp = HawkesNLL_oos_interval(theta_ols_tmp, ...
                            fitTimes, testTimesLocal, ...
                            predStartLocal, predEndLocal, n);
                        
                        aic_oos_tmp = 2*nParam + 2*nll_oos_tmp;
                        
                        NLL_OLS_OOS_all(jj,m) = nll_oos_tmp;
                        AIC_OLS_OOS_all(jj,m) = aic_oos_tmp;
                        
                        SE_OLS_all(jj,m)      = (realizedCount - pred_tmp)^2;
                        AE_OLS_all(jj,m)      = abs(realizedCount - pred_tmp);
                        
                        SE_OLS_STAT_all(jj,m) = (realizedCount - pred_stat_tmp)^2;
                        AE_OLS_STAT_all(jj,m) = abs(realizedCount - pred_stat_tmp);
                    end
                end
                
                MSE_MLE  = mean(SE_MLE_all, 'omitnan');
                RMSE_MLE = sqrt(MSE_MLE);
                MAE_MLE  = mean(AE_MLE_all, 'omitnan');
                
                MSE_MLE_STAT  = mean(SE_MLE_STAT_all, 'omitnan');
                RMSE_MLE_STAT = sqrt(MSE_MLE_STAT);
                MAE_MLE_STAT  = mean(AE_MLE_STAT_all, 'omitnan');
                
                MSE_OLS  = mean(SE_OLS_all, 1, 'omitnan');
                RMSE_OLS = sqrt(MSE_OLS);
                MAE_OLS  = mean(AE_OLS_all, 1, 'omitnan');
                
                MSE_OLS_STAT  = mean(SE_OLS_STAT_all, 1, 'omitnan');
                RMSE_OLS_STAT = sqrt(MSE_OLS_STAT);
                MAE_OLS_STAT  = mean(AE_OLS_STAT_all, 1, 'omitnan');
                
                NLL_MLE_IS  = mean(NLL_MLE_IS_all, 'omitnan');
                AIC_MLE_IS  = mean(AIC_MLE_IS_all, 'omitnan');
                NLL_MLE_OOS = mean(NLL_MLE_OOS_all, 'omitnan');
                AIC_MLE_OOS = mean(AIC_MLE_OOS_all, 'omitnan');
                
                OLSLOSS_IS  = mean(OLSLOSS_IS_all, 1, 'omitnan');
                
                NLL_OLS_IS  = mean(NLL_OLS_IS_all, 1, 'omitnan');
                AIC_OLS_IS  = mean(AIC_OLS_IS_all, 1, 'omitnan');
                NLL_OLS_OOS = mean(NLL_OLS_OOS_all, 1, 'omitnan');
                AIC_OLS_OOS = mean(AIC_OLS_OOS_all, 1, 'omitnan');
                
                rr = rr + 1;
                
                results(rr).file = fileName;
                results(rr).n = n;
                results(rr).N = N;
                results(rr).L_min = Lmin;
                results(rr).L_sec = Lsec;
                results(rr).nForecast = nForecast;
                results(rr).grid_val = grid_val(:).';
                
                results(rr).MSE_MLE = MSE_MLE;
                results(rr).RMSE_MLE = RMSE_MLE;
                results(rr).MAE_MLE = MAE_MLE;
                
                results(rr).MSE_MLE_STAT = MSE_MLE_STAT;
                results(rr).RMSE_MLE_STAT = RMSE_MLE_STAT;
                results(rr).MAE_MLE_STAT = MAE_MLE_STAT;
                
                results(rr).MSE_OLS = MSE_OLS;
                results(rr).RMSE_OLS = RMSE_OLS;
                results(rr).MAE_OLS = MAE_OLS;
                
                results(rr).MSE_OLS_STAT = MSE_OLS_STAT;
                results(rr).RMSE_OLS_STAT = RMSE_OLS_STAT;
                results(rr).MAE_OLS_STAT = MAE_OLS_STAT;
                
                results(rr).NLL_MLE_IS = NLL_MLE_IS;
                results(rr).AIC_MLE_IS = AIC_MLE_IS;
                results(rr).NLL_MLE_OOS = NLL_MLE_OOS;
                results(rr).AIC_MLE_OOS = AIC_MLE_OOS;
                
                results(rr).OLSLOSS_IS = OLSLOSS_IS;
                results(rr).OLSLOSS_IS_all = OLSLOSS_IS_all;
                
                results(rr).NLL_OLS_IS = NLL_OLS_IS;
                results(rr).AIC_OLS_IS = AIC_OLS_IS;
                results(rr).NLL_OLS_OOS = NLL_OLS_OOS;
                results(rr).AIC_OLS_OOS = AIC_OLS_OOS;
                
                results(rr).SE_MLE_all = SE_MLE_all;
                results(rr).AE_MLE_all = AE_MLE_all;
                results(rr).SE_MLE_STAT_all = SE_MLE_STAT_all;
                results(rr).AE_MLE_STAT_all = AE_MLE_STAT_all;
                
                results(rr).SE_OLS_all = SE_OLS_all;
                results(rr).AE_OLS_all = AE_OLS_all;
                results(rr).SE_OLS_STAT_all = SE_OLS_STAT_all;
                results(rr).AE_OLS_STAT_all = AE_OLS_STAT_all;
                
                results(rr).NLL_MLE_IS_all = NLL_MLE_IS_all;
                results(rr).AIC_MLE_IS_all = AIC_MLE_IS_all;
                results(rr).NLL_MLE_OOS_all = NLL_MLE_OOS_all;
                results(rr).AIC_MLE_OOS_all = AIC_MLE_OOS_all;
                
                results(rr).NLL_OLS_IS_all = NLL_OLS_IS_all;
                results(rr).AIC_OLS_IS_all = AIC_OLS_IS_all;
                results(rr).NLL_OLS_OOS_all = NLL_OLS_OOS_all;
                results(rr).AIC_OLS_OOS_all = AIC_OLS_OOS_all;
                
                results(rr).pred_MLE_all = pred_MLE_all;
                results(rr).pred_MLE_STAT_all = pred_MLE_STAT_all;
                results(rr).pred_OLS_all = pred_OLS_all;
                results(rr).pred_OLS_STAT_all = pred_OLS_STAT_all;
                results(rr).realized_all = realized_all;
                results(rr).dtGrid_all = dtGrid_all;
                
                results(rr).mu_MLE_store = mu_MLE_store;
                results(rr).alpha_MLE_store = alpha_MLE_store;
                results(rr).beta_MLE_store = beta_MLE_store;
                results(rr).eta_MLE_store = eta_MLE_store;
                
                results(rr).mu_OLS_store = mu_OLS_store;
                results(rr).alpha_OLS_store = alpha_OLS_store;
                results(rr).eta_OLS_store = eta_OLS_store;
                
                results(rr).status = 'ok';
                results(rr).message = '';
                
                save_compact_results(matFile, results, grid_val, n, ...
                    assetName, L_list_min, L_list_sec);
                
                fprintf(fid, '%-*s', fileW, fileName);
                fprintf(fid, '%*d', intW, Lmin);
                fprintf(fid, '%*d', intW, N);
                fprintf(fid, '%*d', intW, nForecast);
                
                fprintf(fid, '%*.10g', numW, MSE_MLE);
                fprintf(fid, '%*.10g', numW, RMSE_MLE);
                fprintf(fid, '%*.10g', numW, MAE_MLE);
                
                fprintf(fid, '%*.10g', numW, MSE_MLE_STAT);
                fprintf(fid, '%*.10g', numW, RMSE_MLE_STAT);
                fprintf(fid, '%*.10g', numW, MAE_MLE_STAT);
                
                fprintf(fid, '%*.10g', numW, NLL_MLE_IS);
                fprintf(fid, '%*.10g', numW, AIC_MLE_IS);
                fprintf(fid, '%*.10g', numW, NLL_MLE_OOS);
                fprintf(fid, '%*.10g', numW, AIC_MLE_OOS);
                
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, MSE_OLS(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, RMSE_OLS(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, MAE_OLS(m));
                end
                
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, MSE_OLS_STAT(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, RMSE_OLS_STAT(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, MAE_OLS_STAT(m));
                end
                
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, OLSLOSS_IS(m));
                end
                
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, NLL_OLS_IS(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, AIC_OLS_IS(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, NLL_OLS_OOS(m));
                end
                for m = 1:Mgrid
                    fprintf(fid, '%*.10g', numW, AIC_OLS_OOS(m));
                end
                
                fprintf(fid, '\n');
                drawnow;
                
                fprintf('Saved compact MAT: %s | L=%d min | rr=%d\n', fileName, Lmin, rr);
            end
            
        catch ME
            warning('n=%d failed on file %s\nReason: %s', ...
                n, fileName, ME.message);
        end
    end
    
    fclose(fid);
    
    save_compact_results(matFile, results, grid_val, n, ...
        assetName, L_list_min, L_list_sec);
    
    fprintf('\nDone for n=%d.\n', n);
    fprintf('TXT file saved in:\n%s\n', outFile);
    fprintf('Compact MAT file saved in:\n%s\n', matFile);
    
    clear cleanupObj;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Asset specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following code upload all the results generated by the previous section 
dataFolder  = fullfile(pwd, 'Data', 'ASSET'); 
filePattern = 'taqASSET-*.csv';
assetName   = 'AssetName';

% Actual number of kernels in the saved files
nKernel_list = 2:5;

figFolder = fullfile(dataFolder, 'figures_prediction');
if ~exist(figFolder, 'dir')
    mkdir(figFolder);
end

grid_val = logspace(-6.5,2,96).';
plotIdx = 1:length(grid_val);

% Marker rules for overlapping MLE curves only
maeCloseThreshold  = 0.2;
rmseCloseThreshold = 2.0;

ALL = struct([]);
cc = 0;

for ii = 1:numel(nKernel_list)
    
    nKernel = nKernel_list(ii);
    n       = nKernel - 1;
    
    matFile = fullfile(dataFolder, ...
        sprintf('hawkes_prediction_%s_n%d_rolling2h_from1300_allbins.mat', ...
        assetName, nKernel));
    
    if ~isfile(matFile)
        warning('Missing file for n=%d: %s', n, matFile);
        continue;
    end
    
    S = load(matFile);
    
    cc = cc + 1;
    ALL(cc).nKernel  = nKernel;
    ALL(cc).n        = n;
    ALL(cc).results  = S.results;
    ALL(cc).grid_val = S.grid_val(:);
    
    fprintf('Loaded n=%d: %s\n', n, matFile);
end

if isempty(ALL)
    error('No MAT files were loaded.');
end

L_all = [];
for kk = 1:numel(ALL)
    L_all = [L_all; [ALL(kk).results.L_min]'];
end
L_list = unique(L_all);

for ll = 1:numel(L_list)
    
    Lmin = L_list(ll);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MAE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Visible','off'); hold on;
    leg = {};
    
    MAE_OLS_store = cell(numel(ALL),1);
    MAE_MLE_store = cell(numel(ALL),1);
    grid_store    = cell(numel(ALL),1);
    pidx_store    = cell(numel(ALL),1);
    col_store     = cell(numel(ALL),1);
    n_store       = NaN(numel(ALL),1);
    mleBase       = NaN(numel(ALL),1);
    
    for kk = 1:numel(ALL)
        
        n        = ALL(kk).n;
        results  = ALL(kk).results;
        grid_val = ALL(kk).grid_val(:);
        Mgrid    = numel(grid_val);
        
        col = color_for_n(n);
        
        idx  = find([results.L_min]' == Lmin);
        pidx = plotIdx(plotIdx >= 1 & plotIdx <= Mgrid);
        
        AE_OLS_all = cat(1, results(idx).AE_OLS_all);
        AE_MLE_all = cat(1, results(idx).AE_MLE_all);
        
        pred_OLS_all = cat(1, results(idx).pred_OLS_all);
        pred_MLE_all = cat(1, results(idx).pred_MLE_all);
        Y_all        = cat(1, results(idx).realized_all);
        
        MAE_OLS = NaN(1, Mgrid);
        MAE_MLE = NaN(1, Mgrid);
        
        nTotal  = NaN(1, Mgrid);
        nKeep   = NaN(1, Mgrid);
        nDelete = NaN(1, Mgrid);
        
        for m = 1:Mgrid
            
            keep = isfinite(Y_all) ...
                & isfinite(pred_MLE_all) ...
                & isfinite(pred_OLS_all(:,m)) ...
                & pred_MLE_all >= 0 ...
                & pred_OLS_all(:,m) >= 0 ...
                & pred_MLE_all <= explosiveCutoff ...
                & pred_OLS_all(:,m) <= explosiveCutoff ...
                & isfinite(AE_MLE_all) ...
                & isfinite(AE_OLS_all(:,m));
            
            nTotal(m)  = numel(keep);
            nKeep(m)   = sum(keep);
            nDelete(m) = nTotal(m) - nKeep(m);
            
            MAE_OLS(m) = mean(AE_OLS_all(keep,m), 'omitnan');
            MAE_MLE(m) = mean(AE_MLE_all(keep), 'omitnan');
        end
        
        fprintf('\n');
        fprintf('n=%d, L=%d min\n', n, Lmin);
        fprintf('Deleted forecasts:\n');
        fprintf('   min = %d\n', min(nDelete));
        fprintf('   max = %d\n', max(nDelete));
        fprintf('   mean = %.2f\n', mean(nDelete));
        fprintf('   median = %.2f\n', median(nDelete));
        fprintf('\n');
        
        MAE_OLS_store{kk} = MAE_OLS;
        MAE_MLE_store{kk} = MAE_MLE;
        grid_store{kk}    = grid_val;
        pidx_store{kk}    = pidx;
        col_store{kk}     = col;
        n_store(kk)       = n;
        mleBase(kk)       = mean(MAE_MLE(pidx), 'omitnan');
    end
    
    mleMarkerFlag = marker_flags_for_smaller_n(mleBase, n_store, maeCloseThreshold);
    
    for kk = 1:numel(ALL)
        
        MAE_OLS = MAE_OLS_store{kk};
        MAE_MLE = MAE_MLE_store{kk};
        grid_val = grid_store{kk};
        pidx = pidx_store{kk};
        col = col_store{kk};
        n = n_store(kk);
        
        plot(grid_val(pidx), MAE_OLS(pidx), ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        if mleMarkerFlag(kk)
            plot(grid_val(pidx), MAE_MLE(pidx), ...
                '--', ...
                'Color', col, ...
                'LineWidth', 1.5, ...
                'Marker', 'x', ...
                'MarkerIndices', round(linspace(1,numel(pidx),8)));
        else
            plot(grid_val(pidx), MAE_MLE(pidx), ...
                '--', ...
                'Color', col, ...
                'LineWidth', 1.5);
        end
        
        leg{end+1} = sprintf('OLS, n=%d', n);
        leg{end+1} = sprintf('MLE, n=%d', n);
    end
    
    add_y_margin(gca, 0.10);
    
    set(gca,'XScale','log');
    grid on;
    xlabel('Grid parameter $c$', 'Interpreter','latex');
    ylabel('Average MAE', 'Interpreter','latex');
    lgd = legend(leg,'Location','best','Interpreter','latex');
    lgd.FontSize = 8;
    
    exportgraphics(fig, fullfile(figFolder, ...
        sprintf('Prediction_MAE_%s_alln_L%dmin.pdf', ...
        assetName, Lmin)), ...
        'ContentType','vector');
    close(fig);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MSE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Visible','off'); hold on;
    leg = {};
    
    for kk = 1:numel(ALL)
        
        n        = ALL(kk).n;
        results  = ALL(kk).results;
        grid_val = ALL(kk).grid_val(:);
        Mgrid    = numel(grid_val);
        
        col = color_for_n(n);
        
        idx  = find([results.L_min]' == Lmin);
        pidx = plotIdx(plotIdx >= 1 & plotIdx <= Mgrid);
        
        SE_OLS_all = cat(1, results(idx).SE_OLS_all);
        SE_MLE_all = cat(1, results(idx).SE_MLE_all);
        
        pred_OLS_all = cat(1, results(idx).pred_OLS_all);
        pred_MLE_all = cat(1, results(idx).pred_MLE_all);
        Y_all        = cat(1, results(idx).realized_all);
        
        MSE_OLS = NaN(1, Mgrid);
        MSE_MLE = NaN(1, Mgrid);
        
        for m = 1:Mgrid
            
            keep = isfinite(Y_all) ...
                & isfinite(pred_MLE_all) ...
                & isfinite(pred_OLS_all(:,m)) ...
                & pred_MLE_all >= 0 ...
                & pred_OLS_all(:,m) >= 0 ...
                & pred_MLE_all <= explosiveCutoff ...
                & pred_OLS_all(:,m) <= explosiveCutoff ...
                & isfinite(SE_MLE_all) ...
                & isfinite(SE_OLS_all(:,m));
            
            MSE_OLS(m) = mean(SE_OLS_all(keep,m), 'omitnan');
            MSE_MLE(m) = mean(SE_MLE_all(keep), 'omitnan');
        end
        
        plot(grid_val(pidx), MSE_OLS(pidx), ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        plot(grid_val(pidx), MSE_MLE(pidx), ...
            '--', ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        leg{end+1} = sprintf('OLS, n=%d', n);
        leg{end+1} = sprintf('MLE, n=%d', n);
    end
    
    add_y_margin(gca, 0.10);
    
    set(gca,'XScale','log');
    grid on;
    xlabel('Grid parameter $c$', 'Interpreter','latex');
    ylabel('Average MSE', 'Interpreter','latex');
    lgd = legend(leg,'Location','best','Interpreter','latex');
    lgd.FontSize = 8;
    
    exportgraphics(fig, fullfile(figFolder, ...
        sprintf('Prediction_MSE_%s_alln_L%dmin.pdf', ...
        assetName, Lmin)), ...
        'ContentType','vector');
    close(fig);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % RMSE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Visible','off'); hold on;
    leg = {};
    
    RMSE_OLS_store = cell(numel(ALL),1);
    RMSE_MLE_store = cell(numel(ALL),1);
    grid_store     = cell(numel(ALL),1);
    pidx_store     = cell(numel(ALL),1);
    col_store      = cell(numel(ALL),1);
    n_store        = NaN(numel(ALL),1);
    mleBase        = NaN(numel(ALL),1);
    
    for kk = 1:numel(ALL)
        
        n        = ALL(kk).n;
        results  = ALL(kk).results;
        grid_val = ALL(kk).grid_val(:);
        Mgrid    = numel(grid_val);
        
        col = color_for_n(n);
        
        idx  = find([results.L_min]' == Lmin);
        pidx = plotIdx(plotIdx >= 1 & plotIdx <= Mgrid);
        
        SE_OLS_all = cat(1, results(idx).SE_OLS_all);
        SE_MLE_all = cat(1, results(idx).SE_MLE_all);
        
        pred_OLS_all = cat(1, results(idx).pred_OLS_all);
        pred_MLE_all = cat(1, results(idx).pred_MLE_all);
        Y_all        = cat(1, results(idx).realized_all);
        
        RMSE_OLS = NaN(1, Mgrid);
        RMSE_MLE = NaN(1, Mgrid);
        
        for m = 1:Mgrid
            
            keep = isfinite(Y_all) ...
                & isfinite(pred_MLE_all) ...
                & isfinite(pred_OLS_all(:,m)) ...
                & pred_MLE_all >= 0 ...
                & pred_OLS_all(:,m) >= 0 ...
                & pred_MLE_all <= explosiveCutoff ...
                & pred_OLS_all(:,m) <= explosiveCutoff ...
                & isfinite(SE_MLE_all) ...
                & isfinite(SE_OLS_all(:,m));
            
            RMSE_OLS(m) = sqrt(mean(SE_OLS_all(keep,m), 'omitnan'));
            RMSE_MLE(m) = sqrt(mean(SE_MLE_all(keep), 'omitnan'));
        end
        
        RMSE_OLS_store{kk} = RMSE_OLS;
        RMSE_MLE_store{kk} = RMSE_MLE;
        grid_store{kk}     = grid_val;
        pidx_store{kk}     = pidx;
        col_store{kk}      = col;
        n_store(kk)        = n;
        mleBase(kk)        = mean(RMSE_MLE(pidx), 'omitnan');
    end
    
    mleMarkerFlag = marker_flags_for_smaller_n(mleBase, n_store, rmseCloseThreshold);
    
    for kk = 1:numel(ALL)
        
        RMSE_OLS = RMSE_OLS_store{kk};
        RMSE_MLE = RMSE_MLE_store{kk};
        grid_val = grid_store{kk};
        pidx = pidx_store{kk};
        col = col_store{kk};
        n = n_store(kk);
        
        plot(grid_val(pidx), RMSE_OLS(pidx), ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        if mleMarkerFlag(kk)
            plot(grid_val(pidx), RMSE_MLE(pidx), ...
                '--', ...
                'Color', col, ...
                'LineWidth', 1.5, ...
                'Marker', 'x', ...
                'MarkerIndices', round(linspace(1,numel(pidx),8)));
        else
            plot(grid_val(pidx), RMSE_MLE(pidx), ...
                '--', ...
                'Color', col, ...
                'LineWidth', 1.5);
        end
        
        leg{end+1} = sprintf('OLS, n=%d', n);
        leg{end+1} = sprintf('MLE, n=%d', n);
    end
    
    add_y_margin(gca, 0.10);
    
    set(gca,'XScale','log');
    grid on;
    xlabel('Grid parameter $c$', 'Interpreter','latex');
    ylabel('Average RMSE', 'Interpreter','latex');
    lgd = legend(leg,'Location','best','Interpreter','latex');
    lgd.FontSize = 8;
    
    exportgraphics(fig, fullfile(figFolder, ...
        sprintf('Prediction_RMSE_%s_alln_L%dmin.pdf', ...
        assetName, Lmin)), ...
        'ContentType','vector');
    close(fig);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % In-sample AIC
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Visible','off'); hold on;
    leg = {};
    
    for kk = 1:numel(ALL)
        
        n        = ALL(kk).n;
        results  = ALL(kk).results;
        grid_val = ALL(kk).grid_val(:);
        Mgrid    = numel(grid_val);
        
        col = color_for_n(n);
        
        idx  = find([results.L_min]' == Lmin);
        pidx = plotIdx(plotIdx >= 1 & plotIdx <= Mgrid);
        
        AIC_OLS_IS_all = cat(1, results(idx).AIC_OLS_IS_all);
        AIC_MLE_IS_all = cat(1, results(idx).AIC_MLE_IS_all);
        
        pred_OLS_all = cat(1, results(idx).pred_OLS_all);
        pred_MLE_all = cat(1, results(idx).pred_MLE_all);
        Y_all        = cat(1, results(idx).realized_all);
        
        AIC_OLS_IS = NaN(1, Mgrid);
        AIC_MLE_IS = NaN(1, Mgrid);
        
        for m = 1:Mgrid
            
            keep = isfinite(Y_all) ...
                & isfinite(pred_MLE_all) ...
                & isfinite(pred_OLS_all(:,m)) ...
                & pred_MLE_all >= 0 ...
                & pred_OLS_all(:,m) >= 0 ...
                & pred_MLE_all <= explosiveCutoff ...
                & pred_OLS_all(:,m) <= explosiveCutoff ...
                & isfinite(AIC_MLE_IS_all) ...
                & isfinite(AIC_OLS_IS_all(:,m));
            
            AIC_OLS_IS(m) = mean(AIC_OLS_IS_all(keep,m), 'omitnan');
            AIC_MLE_IS(m) = mean(AIC_MLE_IS_all(keep), 'omitnan');
        end
        
        plot(grid_val(pidx), AIC_OLS_IS(pidx), ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        plot(grid_val(pidx), AIC_MLE_IS(pidx), ...
            '--', ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        leg{end+1} = sprintf('OLS, n=%d', n);
        leg{end+1} = sprintf('MLE, n=%d', n);
    end
    
    add_y_margin(gca, 0.10);
    
    set(gca,'XScale','log');
    grid on;
    xlabel('Grid parameter $c$', 'Interpreter','latex');
    ylabel('Average in-sample AIC', 'Interpreter','latex');
    lgd = legend(leg,'Location','best','Interpreter','latex');
    lgd.FontSize = 8;
    
    exportgraphics(fig, fullfile(figFolder, ...
        sprintf('AIC_IS_%s_alln_L%dmin.pdf', ...
        assetName, Lmin)), ...
        'ContentType','vector');
    close(fig);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % OLS in-sample loss criterion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Visible','off'); hold on;
    leg = {};
    
    for kk = 1:numel(ALL)
        
        n        = ALL(kk).n;
        results  = ALL(kk).results;
        grid_val = ALL(kk).grid_val(:);
        Mgrid    = numel(grid_val);
        
        col = color_for_n(n);
        
        idx  = find([results.L_min]' == Lmin);
        pidx = plotIdx(plotIdx >= 1 & plotIdx <= Mgrid);
        
        OLSLOSS_IS_all = cat(1, results(idx).OLSLOSS_IS_all);
        
        pred_OLS_all = cat(1, results(idx).pred_OLS_all);
        pred_MLE_all = cat(1, results(idx).pred_MLE_all);
        Y_all        = cat(1, results(idx).realized_all);
        
        OLSLOSS_IS = NaN(1, Mgrid);
        
        for m = 1:Mgrid
            
            keep = isfinite(Y_all) ...
                & isfinite(pred_MLE_all) ...
                & isfinite(pred_OLS_all(:,m)) ...
                & pred_MLE_all >= 0 ...
                & pred_OLS_all(:,m) >= 0 ...
                & pred_MLE_all <= explosiveCutoff ...
                & pred_OLS_all(:,m) <= explosiveCutoff ...
                & isfinite(OLSLOSS_IS_all(:,m));
            
            OLSLOSS_IS(m) = mean(OLSLOSS_IS_all(keep,m), 'omitnan');
        end
        
        plot(grid_val(pidx), OLSLOSS_IS(pidx), ...
            'Color', col, ...
            'LineWidth', 1.5);
        
        leg{end+1} = sprintf('OLS loss, n=%d', n);
    end
    
    set(gca,'XScale','log');
    grid on;
    xlabel('Grid parameter $c$', 'Interpreter','latex');
    ylabel('Average OLS in-sample loss', 'Interpreter','latex');
    lgd = legend(leg,'Location','best','Interpreter','latex');
    lgd.FontSize = 8;
    
    exportgraphics(fig, fullfile(figFolder, ...
        sprintf('OLSLOSS_IS_%s_alln_L%dmin.pdf', ...
        assetName, Lmin)), ...
        'ContentType','vector');
    close(fig);
    
    fprintf('Saved all-n figures for L=%d min.\n', Lmin);
end