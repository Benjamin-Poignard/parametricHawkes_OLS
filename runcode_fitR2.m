addpath(genpath(pwd));
clear; clc;

rng(42,'twister');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Asset specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code upload the data through their csv files, located in the folder 'Data'
% All the data are in csv format, and each csv file represents 1 trading
% day. All csv files are included in the subfolder 'Data/ASSET' 
dataFolder  = fullfile(pwd, 'Data', 'ASSET'); 
filePattern = 'taqASSET-*.csv';
assetName   = 'AssetName';

n_list = 2:5;
n_ref = 3;

sessionStart = duration(9,30,0);
sessionEnd   = duration(15,30,0);
Tfit = seconds(sessionEnd - sessionStart);

grid_val_cand = logspace(-6.5,2,120).';
grid_idx_list = [60 65 70 75 80 85 90 95 100];
grid_c_list = grid_val_cand(grid_idx_list);

files = dir(fullfile(dataFolder,filePattern));
[~,idx] = sort({files.name});
files = files(idx);

options = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'MaxIterations',50000, ...
    'MaxFunctionEvaluations',2e5, ...
    'SpecifyObjectiveGradient',false);

nFiles = numel(files);
nK = numel(n_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First pass: estimate daily MLE with n_ref = 3
% Used only for original fixed-bin normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_max_daily = NaN(nFiles,1);

parfor ii = 1:nFiles
    
    rng(42 + ii,'twister');
    
    fileName = files(ii).name;
    filePath = fullfile(files(ii).folder,fileName);
    
    try
        tbl = readtable(filePath);
        
        if ~ismember('ts',tbl.Properties.VariableNames)
            error('Column "ts" not found.');
        end
        
        ts = tbl.ts;
        
        if ~isduration(ts)
            try
                ts = duration(string(ts),'InputFormat','hh:mm:ss.SSS');
            catch
                ts = duration(string(ts),'InputFormat','hh:mm:ss');
            end
        end
        
        ts = ts(ts >= sessionStart & ts <= sessionEnd);
        
        fitTimes = seconds(ts - sessionStart);
        fitTimes = clean_event_times(fitTimes);
        Nfit = numel(fitTimes);
        
        if Nfit == 0
            continue;
        end
        
        fitMarks = ones(size(fitTimes));
        
        mu0    = max(Nfit/Tfit*0.2,1e-3);
        alpha0 = (-2 + 4*rand(n_ref,1)) + 20*ones(n_ref,1);
        beta0  = (-2 + 4*rand(n_ref,1)) + 60*ones(n_ref,1);
        theta0 = [mu0; alpha0; beta0];
        
        obj = @(x) HawkesMLE_full(x,fitTimes,fitMarks,Tfit,n_ref);
        
        theta_ref = fmincon(obj,theta0,[],[],[],[],[],[], ...
            @(x) constr_hawkes_full(x,n_ref),options);
        
        beta_ref = theta_ref(2+n_ref:end);
        beta_max_daily(ii) = max(beta_ref);
        
    catch ME
        fprintf('Reference MLE failed %s | %s\n',fileName,ME.message);
    end
end

average_max = mean(beta_max_daily,'omitnan');

% Original fixed-bin grid
grid_val_list = grid_c_list / average_max;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over selected grid values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for gg = 1:numel(grid_idx_list)
    
    grid_idx = grid_idx_list(gg);
    c_raw = grid_c_list(gg);
    
    % Original fixed bin using average_max
    dt_fixed = c_raw / average_max;
    
    R2_daily    = NaN(nFiles,nK);
    AIC_daily   = NaN(nFiles,nK);
    NLL_daily   = NaN(nFiles,nK);
    NLL0_daily  = NaN(nFiles,1);
    
    % Alternative daily beta-normalized bin
    R2_alt_daily   = NaN(nFiles,nK);
    AIC_alt_daily  = NaN(nFiles,nK);
    NLL_alt_daily  = NaN(nFiles,nK);
    dt_alt_daily   = NaN(nFiles,nK);
    
    beta_mle_max_daily_n = NaN(nFiles,nK);
    eta_mle_daily        = NaN(nFiles,nK);
    eta_ols_fixed_daily  = NaN(nFiles,nK);
    eta_ols_alt_daily    = NaN(nFiles,nK);
    
    results_cell = cell(nFiles,1);
    
    parfor ii = 1:nFiles
        
        rng(42 + 100000*gg + ii,'twister');
        
        fileName = files(ii).name;
        filePath = fullfile(files(ii).folder,fileName);
        
        R2_row  = NaN(1,nK);
        AIC_row = NaN(1,nK);
        NLL_row = NaN(1,nK);
        
        R2_alt_row  = NaN(1,nK);
        AIC_alt_row = NaN(1,nK);
        NLL_alt_row = NaN(1,nK);
        dt_alt_row  = NaN(1,nK);
        
        beta_max_row = NaN(1,nK);
        eta_mle_row = NaN(1,nK);
        eta_ols_fixed_row = NaN(1,nK);
        eta_ols_alt_row = NaN(1,nK);
        
        NLL0 = NaN;
        message_row = strings(1,nK);
        message_alt_row = strings(1,nK);
        
        try
            tbl = readtable(filePath);
            
            if ~ismember('ts',tbl.Properties.VariableNames)
                error('Column "ts" not found.');
            end
            
            ts = tbl.ts;
            
            if ~isduration(ts)
                try
                    ts = duration(string(ts),'InputFormat','hh:mm:ss.SSS');
                catch
                    ts = duration(string(ts),'InputFormat','hh:mm:ss');
                end
            end
            
            ts = ts(ts >= sessionStart & ts <= sessionEnd);
            
            fitTimes = seconds(ts - sessionStart);
            fitTimes = clean_event_times(fitTimes);
            Nfit = numel(fitTimes);
            
            if Nfit == 0
                error('No valid events in 09:30--15:30 window.');
            end
            
            fitMarks = ones(size(fitTimes));
            
            mu0_null = Nfit/Tfit;
            NLL0 = mu0_null*Tfit - Nfit*log(mu0_null);
            
            variable_fixed = bin_events(fitTimes,Tfit,dt_fixed);
            
            for kk = 1:nK
                
                n = n_list(kk);
                nParam = 2*n + 1;
                
                try
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % MLE for this day and this n
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    mu0    = max(Nfit/Tfit*0.2,1e-3);
                    alpha0 = (-2 + 4*rand(n,1)) + 20*ones(n,1);
                    beta0  = (-2 + 4*rand(n,1)) + 60*ones(n,1);
                    theta0 = [mu0; alpha0; beta0];
                    
                    obj = @(x) HawkesMLE_full(x,fitTimes,fitMarks,Tfit,n);
                    
                    theta_mle = fmincon(obj,theta0,[],[],[],[],[],[], ...
                        @(x) constr_hawkes_full(x,n),options);
                    
                    mu_mle = theta_mle(1);
                    alpha_mle = theta_mle(2:1+n);
                    beta_mle = theta_mle(2+n:end);
                    
                    [beta_mle,ord] = sort(beta_mle(:));
                    alpha_mle = alpha_mle(ord);
                    
                    beta_max_row(kk) = max(beta_mle);
                    eta_mle_row(kk) = sum(alpha_mle ./ beta_mle);
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Original AIC: fixed dt = c_raw / average_max
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if ~isfinite(dt_fixed) || dt_fixed <= 0
                        error('Invalid fixed dt.');
                    end
                    
                    est_ols = hawkes_OLS(variable_fixed.nzIdx,variable_fixed.nzVal, ...
                        variable_fixed.T,dt_fixed,beta_mle);
                    
                    theta_ols = est_ols.theta;
                    
                    [XtX,Xty,yTy,Sy] = hawkes_OLS_sufficient_stats_R2( ...
                        variable_fixed.nzIdx,variable_fixed.nzVal, ...
                        variable_fixed.T,dt_fixed,beta_mle);
                    
                    SSE = yTy - 2*theta_ols.'*Xty + theta_ols.'*XtX*theta_ols;
                    SST = yTy - Sy^2/variable_fixed.T;
                    
                    if isfinite(SSE) && isfinite(SST) && SST > 0
                        R2_row(kk) = 1 - SSE/SST;
                    end
                    
                    theta_ols_full = [theta_ols(1); theta_ols(2:end); beta_mle];
                    
                    nll_ols = HawkesMLE_full(theta_ols_full,fitTimes,fitMarks,Tfit,n);
                    aic_ols = 2*nParam + 2*nll_ols;
                    
                    NLL_row(kk) = nll_ols;
                    AIC_row(kk) = aic_ols;
                    eta_ols_fixed_row(kk) = sum(theta_ols(2:end) ./ beta_mle);
                    
                    message_row(kk) = "ok";
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Alternative AIC:
                    % daily dt_alt = c_raw / max(beta_mle)
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    dt_alt = c_raw / max(beta_mle);
                    dt_alt_row(kk) = dt_alt;
                    
                    if ~isfinite(dt_alt) || dt_alt <= 0
                        error('Invalid alternative daily dt.');
                    end
                    
                    variable_alt = bin_events(fitTimes,Tfit,dt_alt);
                    
                    est_ols_alt = hawkes_OLS(variable_alt.nzIdx,variable_alt.nzVal, ...
                        variable_alt.T,dt_alt,beta_mle);
                    
                    theta_ols_alt = est_ols_alt.theta;
                    
                    [XtX_alt,Xty_alt,yTy_alt,Sy_alt] = hawkes_OLS_sufficient_stats_R2( ...
                        variable_alt.nzIdx,variable_alt.nzVal, ...
                        variable_alt.T,dt_alt,beta_mle);
                    
                    SSE_alt = yTy_alt - 2*theta_ols_alt.'*Xty_alt ...
                        + theta_ols_alt.'*XtX_alt*theta_ols_alt;
                    
                    SST_alt = yTy_alt - Sy_alt^2/variable_alt.T;
                    
                    if isfinite(SSE_alt) && isfinite(SST_alt) && SST_alt > 0
                        R2_alt_row(kk) = 1 - SSE_alt/SST_alt;
                    end
                    
                    theta_ols_alt_full = [theta_ols_alt(1); theta_ols_alt(2:end); beta_mle];
                    
                    nll_ols_alt = HawkesMLE_full(theta_ols_alt_full,fitTimes,fitMarks,Tfit,n);
                    aic_ols_alt = 2*nParam + 2*nll_ols_alt;
                    
                    NLL_alt_row(kk) = nll_ols_alt;
                    AIC_alt_row(kk) = aic_ols_alt;
                    eta_ols_alt_row(kk) = sum(theta_ols_alt(2:end) ./ beta_mle);
                    
                    message_alt_row(kk) = "ok";
                    
                catch ME_inner
                    message_row(kk) = string(ME_inner.message);
                    message_alt_row(kk) = string(ME_inner.message);
                    fprintf('cidx=%d | c=%g | FAILED %s | n=%d | %s\n', ...
                        grid_idx,c_raw,fileName,n,ME_inner.message);
                end
            end
            
        catch ME_outer
            fprintf('cidx=%d | c=%g | FAILED %s | reading/preprocessing | %s\n', ...
                grid_idx,c_raw,fileName,ME_outer.message);
            message_row(:) = string(ME_outer.message);
            message_alt_row(:) = string(ME_outer.message);
        end
        
        R2_daily(ii,:)   = R2_row;
        AIC_daily(ii,:)  = AIC_row;
        NLL_daily(ii,:)  = NLL_row;
        NLL0_daily(ii)   = NLL0;
        
        R2_alt_daily(ii,:)  = R2_alt_row;
        AIC_alt_daily(ii,:) = AIC_alt_row;
        NLL_alt_daily(ii,:) = NLL_alt_row;
        dt_alt_daily(ii,:)  = dt_alt_row;
        
        beta_mle_max_daily_n(ii,:) = beta_max_row;
        eta_mle_daily(ii,:)        = eta_mle_row;
        eta_ols_fixed_daily(ii,:)  = eta_ols_fixed_row;
        eta_ols_alt_daily(ii,:)    = eta_ols_alt_row;
        
        res = struct();
        res.file = fileName;
        res.NLL0 = NLL0;
        res.c_raw = c_raw;
        res.grid_idx = grid_idx;
        res.dt_fixed_average_max = dt_fixed;
        
        for kk = 1:nK
            n = n_list(kk);
            
            res.(sprintf('R2_fixed_n%d',n)) = R2_row(kk);
            res.(sprintf('AIC_fixed_n%d',n)) = AIC_row(kk);
            res.(sprintf('NLL_fixed_n%d',n)) = NLL_row(kk);
            
            res.(sprintf('R2_alt_n%d',n)) = R2_alt_row(kk);
            res.(sprintf('AIC_alt_n%d',n)) = AIC_alt_row(kk);
            res.(sprintf('NLL_alt_n%d',n)) = NLL_alt_row(kk);
            res.(sprintf('dt_alt_n%d',n)) = dt_alt_row(kk);
            
            res.(sprintf('betaMax_mle_n%d',n)) = beta_max_row(kk);
            res.(sprintf('eta_mle_n%d',n)) = eta_mle_row(kk);
            res.(sprintf('eta_ols_fixed_n%d',n)) = eta_ols_fixed_row(kk);
            res.(sprintf('eta_ols_alt_n%d',n)) = eta_ols_alt_row(kk);
            
            res.(sprintf('message_fixed_n%d',n)) = message_row(kk);
            res.(sprintf('message_alt_n%d',n)) = message_alt_row(kk);
        end
        
        results_cell{ii} = res;
    end
    
    results = [results_cell{:}];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TXT output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    outTxt = fullfile(dataFolder, ...
        sprintf('hawkes_daily_R2_AIC_OLS_%s_cidx%d_c%.6g.txt', ...
        assetName,grid_idx,c_raw));
    
    fid = fopen(outTxt,'w');
    
    fprintf(fid,'%-30s', 'file');
    
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('R2_fixed_n%d',n_list(kk)));
    end
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('AIC_fixed_n%d',n_list(kk)));
    end
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('NLL_fixed_n%d',n_list(kk)));
    end
    
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('R2_alt_n%d',n_list(kk)));
    end
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('AIC_alt_n%d',n_list(kk)));
    end
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('NLL_alt_n%d',n_list(kk)));
    end
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('dt_alt_n%d',n_list(kk)));
    end
    for kk = 1:nK
        fprintf(fid,'%18s', sprintf('betaMax_n%d',n_list(kk)));
    end
    
    fprintf(fid,'\n');
    
    for ii = 1:nFiles
        fprintf(fid,'%-30s',files(ii).name);
        
        for kk = 1:nK
            fprintf(fid,'%18.10g',R2_daily(ii,kk));
        end
        for kk = 1:nK
            fprintf(fid,'%18.10g',AIC_daily(ii,kk));
        end
        for kk = 1:nK
            fprintf(fid,'%18.10g',NLL_daily(ii,kk));
        end
        
        for kk = 1:nK
            fprintf(fid,'%18.10g',R2_alt_daily(ii,kk));
        end
        for kk = 1:nK
            fprintf(fid,'%18.10g',AIC_alt_daily(ii,kk));
        end
        for kk = 1:nK
            fprintf(fid,'%18.10g',NLL_alt_daily(ii,kk));
        end
        for kk = 1:nK
            fprintf(fid,'%18.10g',dt_alt_daily(ii,kk));
        end
        for kk = 1:nK
            fprintf(fid,'%18.10g',beta_mle_max_daily_n(ii,kk));
        end
        
        fprintf(fid,'\n');
    end
    
    fclose(fid);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MAT output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    outMat = fullfile(dataFolder, ...
        sprintf('hawkes_daily_R2_AIC_OLS_%s_cidx%d_c%.6g.mat', ...
        assetName,grid_idx,c_raw));
    
    save(outMat, ...
        'results', ...
        'R2_daily', ...
        'AIC_daily', ...
        'NLL_daily', ...
        'NLL0_daily', ...
        'R2_alt_daily', ...
        'AIC_alt_daily', ...
        'NLL_alt_daily', ...
        'dt_alt_daily', ...
        'beta_mle_max_daily_n', ...
        'eta_mle_daily', ...
        'eta_ols_fixed_daily', ...
        'eta_ols_alt_daily', ...
        'beta_max_daily', ...
        'average_max', ...
        'grid_val_cand', ...
        'grid_idx_list', ...
        'grid_c_list', ...
        'grid_idx', ...
        'c_raw', ...
        'grid_val_list', ...
        'n_ref', ...
        'n_list', ...
        'dt_fixed', ...
        'assetName', ...
        '-v7.3');
    
    fprintf('\nSaved TXT:\n%s\n',outTxt);
    fprintf('Saved MAT:\n%s\n',outMat);
end

%% Print the results
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assets = {
    'Asset1Name',   fullfile(pwd,'Data','ASSET1/R2_stats')
    'Asset2Name', fullfile(pwd,'Data','ASSET2/R2_stats')
    };
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid_val_cand = logspace(-6.5,2,120).';
grid_idx_list = [60 65 70 75 80 85 90 95 100];
grid_c_list   = grid_val_cand(grid_idx_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for aa = 1:size(assets,1)
    
    assetName = assets{aa,1};
    dataFolder = assets{aa,2};
    
    fprintf('\n=====================================================\n');
    fprintf('Asset: %s\n',assetName);
    fprintf('=====================================================\n');
    
    for gg = 1:numel(grid_idx_list)
        
        grid_idx = grid_idx_list(gg);
        c_raw    = grid_c_list(gg);
        
        matFile = fullfile(dataFolder,...
            sprintf('hawkes_daily_R2_AIC_OLS_%s_cidx%d_c%.6g.mat',...
            assetName,grid_idx,c_raw));
        
        if ~isfile(matFile)
            fprintf('Missing file: %s\n',matFile);
            continue;
        end
        
        S = load(matFile,'R2_daily','AIC_daily');
        
        meanR2  = mean(S.R2_daily,1,'omitnan');
        meanAIC = mean(S.AIC_daily,1,'omitnan');
        
        fprintf('\n');
        fprintf('c-index = %3d    c = %.6g\n',grid_idx,c_raw);
        
        for kk = 1:4
            fprintf(['  n = %d :  mean(R2) = %.8f   ' ...
                'mean(AIC) = %.4f\n'], ...
                kk+1,meanR2(kk),meanAIC(kk));
        end
        
    end
end