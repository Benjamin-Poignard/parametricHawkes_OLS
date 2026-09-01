% Hypothesis testing on the homogeneity assumption of the excitation
% parameters for simulated data.

% Hypothesis testing on homogeneity restrictions for simulated Hawkes data.
% The decay parameters beta are treated as known.

addpath(genpath(pwd));
clear; clc;

rng(42,'twister');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sim = 2000;
T   = 21600;

grid_select = logspace(-6.5,2,120).';
c_grid = grid_select(1:100).';

c0_index_list = [50 60 70];
c0_list = c_grid(c0_index_list);

alphaLevel_list = [0.01, 0.05, 0.10];

mT = round(T^(1/4-0.01));
thetaWeight = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% True Hawkes parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = 5;
B_true = [300; 4000; 15000];
K = numel(B_true);

% True parameter satisfying H0
A_H0 = [100; 1500; 1500];
% True parameter for which H0 is not satisfied
A_H1 = [100; 1500; 2000];

fprintf('BR A_H0 = %.6f\n',sum(A_H0 ./ B_true));
fprintf('BR A_H1 = %.6f\n',sum(A_H1 ./ B_true));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter cases
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramNames = ["A_H0","A_H1"];
A_cases = {A_H0,A_H1};

nParamCases = numel(A_cases);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear restrictions: theta = [mu alpha1 alpha2 alpha3]'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R_cases = cell(3,1);
r_cases = cell(3,1);
R_names = strings(3,1);

R_cases{1} = [0  1 -1  0];
r_cases{1} = 0;
R_names(1) = "a1_equal_a2";

R_cases{2} = [0  1  0 -1];
r_cases{2} = 0;
R_names(2) = "a1_equal_a3";

R_cases{3} = [0  0  1 -1];
r_cases{3} = 0;
R_names(3) = "a2_equal_a3";

nRcases = numel(R_cases);

for iparam = 1:nParamCases
    for ir = 1:nRcases
        restrictionValue = ...
            R_cases{ir} * [mu; A_cases{iparam}] - r_cases{ir};
        
        fprintf('%s | %s | R*[mu;A]-r = %.6f\n', ...
            paramNames(iparam),R_names(ir),restrictionValue);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output folder and files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% User specified folder name where all the results will be saved
resultsFolder = fullfile(pwd,'results_test');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

resultsFile = fullfile(resultsFolder, ...
    'joint_wald_multiR_known_beta_results.mat');

summaryCSV = fullfile(resultsFolder, ...
    'joint_wald_multiR_known_beta_summary.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Monte Carlo simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Results = struct([]);
rr = 0;

for iparam = 1:nParamCases
    
    A_true = A_cases{iparam};
    paramName = paramNames(iparam);
    
    for ig = 1:numel(c0_list)
        
        c0 = c0_list(ig);
        c0_index = c0_index_list(ig);
        
        fprintf('\n=========================================\n');
        fprintf('Parameter case = %s\n',paramName);
        fprintf('Beta           = known\n');
        fprintf('c index        = %d\n',c0_index);
        fprintf('c              = %.8g\n',c0);
        fprintf('BR             = %.6f\n',sum(A_true ./ B_true));
        fprintf('=========================================\n');
        
        S_store       = NaN(Sim,nRcases);
        p_store       = NaN(Sim,nRcases);
        theta_store   = NaN(Sim,K+1);
        reject_store  = false(Sim,nRcases,numel(alphaLevel_list));
        status_store  = false(Sim,1);
        message_store = strings(Sim,1);
        
        parfor s = 1:Sim
            
            rng(42 + 100000*iparam + 1000*ig + s,'twister');
            
            S_s = NaN(1,nRcases);
            p_s = NaN(1,nRcases);
            theta_s = NaN(1,K+1);
            
            reject_s = false(nRcases,numel(alphaLevel_list));
            msg_s = "ok";
            
            try
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Simulate Hawkes process
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [eventsTimes,~,~] = ...
                    simulate_mhawkes(mu,A_true,B_true,T);
                
                % Beta is known
                beta_use = B_true;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Compute OLS estimator and base quantities
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                R_base = eye(K+1);
                r_base = zeros(K+1,1);
                
                optsWald = struct();
                optsWald.waldScale = 'Tbins';
                optsWald.useParallelForLags = false;
                
                out_base = compute_hawkes_ols_wald_statistic( ...
                    eventsTimes,T,c0,beta_use, ...
                    R_base,r_base,mT,thetaWeight,optsWald);
                
                theta_s = out_base.theta_hat(:)';
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Compute each Wald test
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for ir = 1:nRcases
                    
                    Wj = compute_wald_from_base( ...
                        out_base,R_cases{ir},r_cases{ir});
                    
                    S_s(ir) = Wj.S;
                    p_s(ir) = Wj.p;
                    
                    for ilev = 1:numel(alphaLevel_list)
                        reject_s(ir,ilev) = ...
                            isfinite(p_s(ir)) && ...
                            p_s(ir) < alphaLevel_list(ilev);
                    end
                end
                
                status_s = true;
                
            catch ME
                msg_s = string(ME.message);
                status_s = false;
            end
            
            S_store(s,:) = S_s;
            p_store(s,:) = p_s;
            theta_store(s,:) = theta_s;
            reject_store(s,:,:) = reject_s;
            status_store(s) = status_s;
            message_store(s) = msg_s;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Valid replications
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        valid = status_store ...
            & all(isfinite(S_store),2) ...
            & all(isfinite(p_store),2);
        
        rr = rr + 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Store setting information
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Results(rr).paramName = paramName;
        
        Results(rr).c0 = c0;
        Results(rr).c0_index = c0_index;
        Results(rr).ig = ig;
        
        Results(rr).mu = mu;
        Results(rr).A_true = A_true;
        Results(rr).B_true = B_true;
        Results(rr).branchingRatio = sum(A_true ./ B_true);
        
        Results(rr).R_cases = R_cases;
        Results(rr).r_cases = r_cases;
        Results(rr).R_names = R_names;
        Results(rr).nRcases = nRcases;
        
        restrictionValues = NaN(1,nRcases);
        
        for ir = 1:nRcases
            restrictionValues(ir) = ...
                R_cases{ir} * [mu; A_true] - r_cases{ir};
        end
        
        Results(rr).restrictionValues = restrictionValues;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Store Monte Carlo outputs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Results(rr).S = S_store;
        Results(rr).p = p_store;
        Results(rr).theta_hat = theta_store;
        
        Results(rr).reject = reject_store;
        Results(rr).status = status_store;
        Results(rr).valid = valid;
        Results(rr).message = message_store;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Simulation summaries
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Results(rr).nValid = sum(valid);
        
        Results(rr).meanS = ...
            mean(S_store(valid,:),1,'omitnan');
        
        Results(rr).medianS = ...
            median(S_store(valid,:),1,'omitnan');
        
        Results(rr).meanP = ...
            mean(p_store(valid,:),1,'omitnan');
        
        Results(rr).medianP = ...
            median(p_store(valid,:),1,'omitnan');
        
        Results(rr).meanTheta = ...
            mean(theta_store(valid,:),1,'omitnan');
        
        empiricalRejectRate = ...
            NaN(nRcases,numel(alphaLevel_list));
        
        for ir = 1:nRcases
            for ilev = 1:numel(alphaLevel_list)
                
                tmpReject = ...
                    squeeze(reject_store(:,ir,ilev));
                
                empiricalRejectRate(ir,ilev) = ...
                    mean(tmpReject(valid));
            end
        end
        
        Results(rr).empiricalRejectRate = ...
            empiricalRejectRate;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Display current results
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('Valid replications: %d / %d\n', ...
            Results(rr).nValid,Sim);
        
        for ir = 1:nRcases
            
            fprintf('Restriction %s\n',R_names(ir));
            fprintf('Mean Wald S    = %.6f\n', ...
                Results(rr).meanS(ir));
            fprintf('Median p-value = %.6f\n', ...
                Results(rr).medianP(ir));
            
            for ilev = 1:numel(alphaLevel_list)
                fprintf('Reject rate alpha %.2f = %.4f\n', ...
                    alphaLevel_list(ilev), ...
                    empiricalRejectRate(ir,ilev));
            end
        end
        
        % Save after each setting
        save(resultsFile, ...
            'Results', ...
            'Sim','T','mu','B_true', ...
            'A_H0','A_H1','A_cases','paramNames', ...
            'R_cases','r_cases','R_names', ...
            'c0_list','c0_index_list','c_grid', ...
            'alphaLevel_list','mT','thetaWeight', ...
            '-v7.3');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nSettings = numel(Results);
nRows = nSettings*nRcases;

paramCol = strings(nRows,1);
RnameCol = strings(nRows,1);

c0Col = NaN(nRows,1);
c0IndexCol = NaN(nRows,1);
BRCol = NaN(nRows,1);
restrictionCol = NaN(nRows,1);
nValidCol = NaN(nRows,1);

meanSCol = NaN(nRows,1);
medianSCol = NaN(nRows,1);
meanPCol = NaN(nRows,1);
medianPCol = NaN(nRows,1);

rejectRate = NaN(nRows,numel(alphaLevel_list));

row = 0;

for ii = 1:nSettings
    for ir = 1:nRcases
        
        row = row + 1;
        
        paramCol(row) = Results(ii).paramName;
        RnameCol(row) = Results(ii).R_names(ir);
        
        c0Col(row) = Results(ii).c0;
        c0IndexCol(row) = Results(ii).c0_index;
        BRCol(row) = Results(ii).branchingRatio;
        
        restrictionCol(row) = ...
            Results(ii).restrictionValues(ir);
        
        nValidCol(row) = Results(ii).nValid;
        
        meanSCol(row) = Results(ii).meanS(ir);
        medianSCol(row) = Results(ii).medianS(ir);
        meanPCol(row) = Results(ii).meanP(ir);
        medianPCol(row) = Results(ii).medianP(ir);
        
        rejectRate(row,:) = ...
            Results(ii).empiricalRejectRate(ir,:);
    end
end

summaryTable = table( ...
    paramCol,RnameCol,c0Col,c0IndexCol,BRCol, ...
    restrictionCol,nValidCol, ...
    meanSCol,medianSCol,meanPCol,medianPCol, ...
    'VariableNames',{ ...
    'paramName','Rname','c0','c0Index', ...
    'branchingRatio','restrictionValue','nValid', ...
    'meanS','medianS','meanP','medianP'});

for ilev = 1:numel(alphaLevel_list)
    
    variableName = sprintf( ...
        'rejectRate_alpha_%g',alphaLevel_list(ilev));
    
    summaryTable.(variableName) = ...
        rejectRate(:,ilev);
end

summaryTable = sortrows(summaryTable, ...
    {'c0Index','paramName','Rname'});

disp(summaryTable);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final save
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save(resultsFile, ...
    'Results','summaryTable', ...
    'Sim','T','mu','B_true', ...
    'A_H0','A_H1','A_cases','paramNames', ...
    'R_cases','r_cases','R_names', ...
    'c0_list','c0_index_list','c_grid', ...
    'alphaLevel_list','mT','thetaWeight', ...
    '-v7.3');

writetable(summaryTable,summaryCSV);

fprintf('\nResults saved in:\n');
fprintf('  %s\n',resultsFile);
fprintf('  %s\n',summaryCSV);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Post-processing: unadjusted empirical size and power
%
% Beta is known in every simulation setting.
%
% Interpretation:
%   - True restriction  -> empirical size
%   - False restriction -> empirical power
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;

resultsFolder = fullfile(pwd,'results_test');

inputFile = fullfile(resultsFolder, ...
    'joint_wald_multiR_known_beta_results.mat');

if ~isfile(inputFile)
    error('Input file not found: %s',inputFile);
end

loadedData = load(inputFile,'Results','Sim');

Results = loadedData.Results;
Sim = loadedData.Sim;

alphaList = [0.01,0.05,0.10];
alphaMain = 0.05;

tolNull = 1e-10;

nSettings = numel(Results);
nRcases = Results(1).nRcases;
nRows = nSettings*nRcases;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Storage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramNameCol = strings(nRows,1);
c0IndexCol = NaN(nRows,1);
c0Col = NaN(nRows,1);

restrictionCol = strings(nRows,1);
restrictionValueCol = NaN(nRows,1);

truthCol = strings(nRows,1);
measureCol = strings(nRows,1);

nValidCol = NaN(nRows,1);

rejectRate = NaN(nRows,numel(alphaList));

row = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the empirical rejection rates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii = 1:nSettings
    
    p = Results(ii).p;
    
    valid = Results(ii).valid(:) ...
        & all(isfinite(p),2);
    
    if size(p,1) ~= Sim
        warning(['Setting %d has %d replications ', ...
            'instead of Sim = %d.'], ...
            ii,size(p,1),Sim);
    end
    
    pValid = p(valid,:);
    nValid = size(pValid,1);
    
    restrictionValues = ...
        Results(ii).restrictionValues(:).';
    
    for ir = 1:nRcases
        
        row = row + 1;
        
        isTrueNull = ...
            abs(restrictionValues(ir)) <= tolNull;
        
        paramNameCol(row) = ...
            Results(ii).paramName;
        
        c0IndexCol(row) = ...
            Results(ii).c0_index;
        
        c0Col(row) = ...
            Results(ii).c0;
        
        restrictionCol(row) = ...
            Results(ii).R_names(ir);
        
        restrictionValueCol(row) = ...
            restrictionValues(ir);
        
        nValidCol(row) = nValid;
        
        if isTrueNull
            truthCol(row) = "True";
            measureCol(row) = "Empirical size";
        else
            truthCol(row) = "False";
            measureCol(row) = "Empirical power";
        end
        
        for ia = 1:numel(alphaList)
            
            alpha = alphaList(ia);
            
            rejectRate(row,ia) = ...
                mean(pValid(:,ir) < alpha);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Full summary table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
summaryTable = table( ...
    paramNameCol,c0IndexCol,c0Col,restrictionCol, ...
    restrictionValueCol,truthCol,measureCol,nValidCol, ...
    'VariableNames',{ ...
    'paramName','c0Index','c0','restriction', ...
    'restrictionValue','nullIsTrue', ...
    'reportedMeasure','nValid'});

for ia = 1:numel(alphaList)
    
    tag = strrep( ...
        sprintf('%.2f',alphaList(ia)),'.','_');
    
    variableName = ...
        sprintf('rejectRate_alpha_%s',tag);
    
    summaryTable.(variableName) = ...
        rejectRate(:,ia);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Percentage version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
percentTable = summaryTable;

for ia = 1:numel(alphaList)
    
    tag = strrep( ...
        sprintf('%.2f',alphaList(ia)),'.','_');
    
    variableName = ...
        sprintf('rejectRate_alpha_%s',tag);
    
    percentTable.(variableName) = ...
        100*percentTable.(variableName);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compact table at alpha = 0.05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mainIndex = find(alphaList == alphaMain,1);

if isempty(mainIndex)
    error('alphaMain is not contained in alphaList.');
end

mainCol = 100*rejectRate(:,mainIndex);

compactTable = table( ...
    paramNameCol,c0IndexCol,c0Col,restrictionCol, ...
    truthCol,measureCol,nValidCol,mainCol, ...
    'VariableNames',{ ...
    'paramName','c0Index','c0','restriction', ...
    'nullIsTrue','reportedMeasure','nValid', ...
    'rejectionPercent_alpha_0_05'});

compactTable = sortrows(compactTable, ...
    {'c0Index','paramName','restriction'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=========================================================\n');
fprintf('Unadjusted empirical rejection rates: known beta\n');
fprintf('True null  -> empirical size\n');
fprintf('False null -> empirical power\n');
fprintf('=========================================================\n');

disp(compactTable);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputMat = fullfile(resultsFolder, ...
    'joint_wald_multiR_known_beta_postprocess.mat');

outputCSVFull = fullfile(resultsFolder, ...
    'joint_wald_multiR_known_beta_all_levels.csv');

outputCSVCompact = fullfile(resultsFolder, ...
    'joint_wald_multiR_known_beta_alpha_005.csv');

save(outputMat, ...
    'summaryTable','percentTable','compactTable', ...
    'alphaList','alphaMain','tolNull', ...
    '-v7.3');

writetable(percentTable,outputCSVFull);
writetable(compactTable,outputCSVCompact);

fprintf('\nResults saved in:\n');
fprintf('  %s\n',outputMat);
fprintf('  %s\n',outputCSVFull);
fprintf('  %s\n',outputCSVCompact);
