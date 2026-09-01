% Hypothesis testing on the homogeneity assumption of the excitation
% parameters for real data

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

% Number of kernels
K = 3;

% Excludes the first and last 30 minutes
sessionStart = duration(9,30,0);
sessionEnd   = duration(15,30,0);
Tfit = seconds(sessionEnd - sessionStart);

% User specified grid size: test run for different values of c
c_grid = logspace(-6.5,2,120);
c_index_list = [60 70 75 80 85 90 95 100];
c_user = c_grid(c_index_list);

alphaLevel_list = [0.01, 0.05, 0.10, 0.15, 0.20];
nAlpha = numel(alphaLevel_list);

mT = round(Tfit^(1/4-0.01));
thetaWeight = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 restrictions, theta = [mu alpha1 alpha2 alpha3]'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The restrictions are user-specified
R_all = [
    0  1 -1  0
    0  1  0 -1
    0  0  1 -1
    ];

r_all = zeros(3,1);
nTests = size(R_all,1);

testNames = [
    "a1=a2"
    "a1=a3"
    "a2=a3"
    ];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir(fullfile(dataFolder,filePattern));
[~,idx] = sort({files.name});
files = files(idx);
nFiles = numel(files);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options for optimizer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'MaxIterations',50000, ...
    'MaxFunctionEvaluations',2e5, ...
    'SpecifyObjectiveGradient',false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over c grid values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ic = 1:numel(c_user)
    
    c_day = c_user(ic);
    c_index = c_index_list(ic);
    
    fprintf('\n=========================================\n');
    fprintf('Running c index = %d, c = %.10g\n',c_index,c_day);
    fprintf('=========================================\n');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Storage for this c value
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S_daily = NaN(nFiles,nTests);
    p_daily = NaN(nFiles,nTests);
    
    theta_ols_daily = NaN(nFiles,K+1);
    theta_mle_daily = NaN(nFiles,2*K+1);
    beta_mle_daily  = NaN(nFiles,K);
    Delta_daily     = NaN(nFiles,1);
    Nfit_daily      = NaN(nFiles,1);
    
    status_daily  = false(nFiles,1);
    message_daily = strings(nFiles,1);
    
    results_cell = cell(nFiles,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Daily loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    parfor ii = 1:nFiles
        
        rng(100000 + 1000*ic + ii,'twister');
        
        fileName = files(ii).name;
        filePath = fullfile(files(ii).folder,fileName);
        
        S_row = NaN(1,nTests);
        p_row = NaN(1,nTests);
        
        theta_ols_row = NaN(1,K+1);
        theta_mle_row = NaN(1,2*K+1);
        beta_mle_row  = NaN(1,K);
        Delta_row     = NaN;
        Nfit          = NaN;
        
        status = false;
        msg = "ok";
        
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
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Step 1: MLE estimation of the Hawkes parameters
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            mu0    = max(Nfit/Tfit*0.2,1e-3);
            alpha0 = (-2 + 4*rand(K,1)) + 20*ones(K,1);
            beta0  = (-2 + 4*rand(K,1)) + 60*ones(K,1);
            theta0 = [mu0; alpha0; beta0];
            
            obj = @(x) HawkesMLE_full(x,fitTimes,fitMarks,Tfit,K);
            
            theta_mle = fmincon(obj,theta0,[],[],[],[],[],[], ...
                @(x) constr_hawkes_full(x,K),options);
            
            mu_mle    = theta_mle(1);
            alpha_mle = theta_mle(2:1+K);
            beta_mle  = theta_mle(2+K:end);
            
            [beta_mle,ord] = sort(beta_mle(:));
            alpha_mle = alpha_mle(ord);
            
            theta_mle_sorted = [mu_mle; alpha_mle; beta_mle];
            
            theta_mle_row = theta_mle_sorted(:)';
            beta_mle_row  = beta_mle(:)';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Step 2: Compute OLS/HAC/Dhat once for this day/c
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Delta_row = c_day / max(beta_mle);
            
            optsWald = struct();
            optsWald.waldScale = 'Tbins';
            optsWald.useParallelForLags = false;
            
            R_base = eye(K+1);
            r_base = zeros(K+1,1);
            
            out_base = compute_hawkes_ols_wald_statistic( ...
                fitTimes, Tfit, c_day, beta_mle, R_base, r_base, ...
                mT, thetaWeight, optsWald);
            
            theta_ols_row = out_base.theta_hat(:)';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Step 3: Three tests from the same Dhat
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for jj = 1:nTests
                Wj = compute_wald_from_base(out_base,R_all(jj,:),r_all(jj));
                S_row(jj) = Wj.S;
                p_row(jj) = Wj.p;
            end
            
            status = true;
            
        catch ME
            msg = string(ME.message);
            fprintf('FAILED %s | %s\n',fileName,ME.message);
        end
        
        S_daily(ii,:) = S_row;
        p_daily(ii,:) = p_row;
        
        theta_ols_daily(ii,:) = theta_ols_row;
        theta_mle_daily(ii,:) = theta_mle_row;
        beta_mle_daily(ii,:)  = beta_mle_row;
        Delta_daily(ii)       = Delta_row;
        Nfit_daily(ii)        = Nfit;
        
        status_daily(ii)  = status;
        message_daily(ii) = msg;
        
        res = struct();
        res.file = fileName;
        res.status = status;
        res.message = msg;
        res.Nfit = Nfit;
        res.c_index = c_index;
        res.c_user = c_day;
        res.Delta = Delta_row;
        res.theta_ols = theta_ols_row;
        res.theta_mle = theta_mle_row;
        res.beta_mle = beta_mle_row;
        
        for jj = 1:nTests
            res.(sprintf('S_test%d',jj)) = S_row(jj);
            res.(sprintf('p_raw_test%d',jj)) = p_row(jj);
        end
        
        results_cell{ii} = res;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Apply Bonferroni corrections
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p_bonf_within_day = min(nTests * p_daily, 1);
    p_bonf_days_by_test = min(nFiles * p_daily, 1);
    p_bonf_total = min(nFiles * nTests * p_daily, 1);
    
    reject_raw_alpha = false(nFiles,nTests,nAlpha);
    reject_bonf_within_day_alpha = false(nFiles,nTests,nAlpha);
    reject_bonf_days_by_test_alpha = false(nFiles,nTests,nAlpha);
    reject_bonf_total_alpha = false(nFiles,nTests,nAlpha);
    
    for ia = 1:nAlpha
        alphaLevel = alphaLevel_list(ia);
        
        reject_raw_alpha(:,:,ia) = p_daily < alphaLevel;
        reject_bonf_within_day_alpha(:,:,ia) = p_bonf_within_day < alphaLevel;
        reject_bonf_days_by_test_alpha(:,:,ia) = p_bonf_days_by_test < alphaLevel;
        reject_bonf_total_alpha(:,:,ia) = p_bonf_total < alphaLevel;
    end
    
    % Default aliases for alpha = 0.05
    [~,idxAlphaDefault] = min(abs(alphaLevel_list - 0.05));
    
    reject_raw = reject_raw_alpha(:,:,idxAlphaDefault);
    reject_bonf_within_day = reject_bonf_within_day_alpha(:,:,idxAlphaDefault);
    reject_bonf_days_by_test = reject_bonf_days_by_test_alpha(:,:,idxAlphaDefault);
    reject_bonf_total = reject_bonf_total_alpha(:,:,idxAlphaDefault);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Results structs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    results = [results_cell{:}];
    
    for ii = 1:nFiles
        for jj = 1:nTests
            results(ii).(sprintf('pBonfWithinDay_test%d',jj)) = p_bonf_within_day(ii,jj);
            results(ii).(sprintf('pBonfDaysByTest_test%d',jj)) = p_bonf_days_by_test(ii,jj);
            results(ii).(sprintf('pBonfTotal_test%d',jj)) = p_bonf_total(ii,jj);
            
            for ia = 1:nAlpha
                aTag = alpha_tag(alphaLevel_list(ia));
                
                results(ii).(sprintf('rejectRaw_%s_test%d',aTag,jj)) = ...
                    reject_raw_alpha(ii,jj,ia);
                results(ii).(sprintf('rejectBonfWithinDay_%s_test%d',aTag,jj)) = ...
                    reject_bonf_within_day_alpha(ii,jj,ia);
                results(ii).(sprintf('rejectBonfDaysByTest_%s_test%d',aTag,jj)) = ...
                    reject_bonf_days_by_test_alpha(ii,jj,ia);
                results(ii).(sprintf('rejectBonfTotal_%s_test%d',aTag,jj)) = ...
                    reject_bonf_total_alpha(ii,jj,ia);
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Summary table
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fileCol = strings(nFiles,1);
    for ii = 1:nFiles
        fileCol(ii) = string(files(ii).name);
    end
    
    summaryTable = table(fileCol, status_daily, message_daily, Nfit_daily, ...
        Delta_daily, ...
        theta_ols_daily(:,1), theta_ols_daily(:,2), ...
        theta_ols_daily(:,3), theta_ols_daily(:,4), ...
        beta_mle_daily(:,1), beta_mle_daily(:,2), beta_mle_daily(:,3), ...
        'VariableNames', {'file','status','message','Nfit','Delta', ...
        'mu_hat','alpha1_hat','alpha2_hat','alpha3_hat', ...
        'beta1_mle','beta2_mle','beta3_mle'});
    
    for jj = 1:nTests
        summaryTable.(sprintf('S_%d',jj)) = S_daily(:,jj);
        summaryTable.(sprintf('pRaw_%d',jj)) = p_daily(:,jj);
        summaryTable.(sprintf('pBonfWithinDay_%d',jj)) = p_bonf_within_day(:,jj);
        summaryTable.(sprintf('pBonfDaysByTest_%d',jj)) = p_bonf_days_by_test(:,jj);
        summaryTable.(sprintf('pBonfTotal_%d',jj)) = p_bonf_total(:,jj);
        
        for ia = 1:nAlpha
            aTag = alpha_tag(alphaLevel_list(ia));
            
            summaryTable.(sprintf('rejectRaw_%s_%d',aTag,jj)) = ...
                reject_raw_alpha(:,jj,ia);
            summaryTable.(sprintf('rejectBonfWithinDay_%s_%d',aTag,jj)) = ...
                reject_bonf_within_day_alpha(:,jj,ia);
            summaryTable.(sprintf('rejectBonfDaysByTest_%s_%d',aTag,jj)) = ...
                reject_bonf_days_by_test_alpha(:,jj,ia);
            summaryTable.(sprintf('rejectBonfTotal_%s_%d',aTag,jj)) = ...
                reject_bonf_total_alpha(:,jj,ia);
        end
    end
    
    testInfo = table((1:nTests)', testNames, R_all, r_all, ...
        'VariableNames', {'testID','testName','R','r'});
    
    rejectCounts = struct();
    rejectCounts.alphaLevel_list = alphaLevel_list;
    
    for ia = 1:nAlpha
        aTag = alpha_tag(alphaLevel_list(ia));
        
        rejectCounts.(sprintf('raw_%s_byTest',aTag)) = ...
            sum(reject_raw_alpha(:,:,ia),1);
        rejectCounts.(sprintf('BonfWithinDay_%s_byTest',aTag)) = ...
            sum(reject_bonf_within_day_alpha(:,:,ia),1);
        rejectCounts.(sprintf('BonfDaysByTest_%s_byTest',aTag)) = ...
            sum(reject_bonf_days_by_test_alpha(:,:,ia),1);
        rejectCounts.(sprintf('BonfTotal_%s_byTest',aTag)) = ...
            sum(reject_bonf_total_alpha(:,:,ia),1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save one MAT and TXT file for this c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    outMat = fullfile(dataFolder, ...
        sprintf('hawkes_daily_wald_K3_equalities_%s_cidx%d_c%.6g.mat', ...
        assetName,c_index,c_day));
    
    outTxt = fullfile(dataFolder, ...
        sprintf('hawkes_daily_wald_K3_equalities_%s_cidx%d_c%.6g.txt', ...
        assetName,c_index,c_day));
    
    save(outMat, ...
        'results', 'summaryTable', 'testInfo', 'rejectCounts', ...
        'S_daily', 'p_daily', ...
        'p_bonf_within_day', 'p_bonf_days_by_test', 'p_bonf_total', ...
        'reject_raw_alpha', ...
        'reject_bonf_within_day_alpha', ...
        'reject_bonf_days_by_test_alpha', ...
        'reject_bonf_total_alpha', ...
        'reject_raw', 'reject_bonf_within_day', ...
        'reject_bonf_days_by_test', 'reject_bonf_total', ...
        'theta_ols_daily', 'theta_mle_daily', ...
        'beta_mle_daily', 'Delta_daily', ...
        'Nfit_daily', 'status_daily', 'message_daily', ...
        'R_all', 'r_all', 'testNames', ...
        'c_day', 'c_index', 'c_index_list', 'c_user', 'c_grid', ...
        'assetName', 'dataFolder', 'filePattern', ...
        'sessionStart', 'sessionEnd', 'Tfit', ...
        'K', 'mT', 'thetaWeight', ...
        'alphaLevel_list', 'nAlpha', 'idxAlphaDefault', ...
        '-v7.3');
    
    writetable(summaryTable,outTxt,'Delimiter','\t');
    
    fprintf('\nSaved MAT file:\n%s\n',outMat);
    fprintf('Saved TXT file:\n%s\n',outTxt);
end
