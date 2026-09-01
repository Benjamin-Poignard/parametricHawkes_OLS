% Simulations (generating QQ plots) for the standardized OLS estimator

addpath(genpath(pwd));
clear; clc;

rng(42,'twister');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 10000; Nsim = 2000;

% User specified grid for bin size
grid_all = logspace(-6.5,2,120).';
c_grid = grid_all;
c_user = grid_all(60);

mT = round(T^(1/4-0.01));
thetaWeight = 0.01;
K = 3;

% User specified folder where to save the results
results_dir = fullfile(pwd,'results_dist');

if ~exist(results_dir,'dir')
    mkdir(results_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% True parameter settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramCases = struct([]);
% Case 1: BR = 0.98, non-capped beta, heterogeneous
% A./B = [0.18; 0.31; 0.49]
paramCases(1).mu = 1;
paramCases(1).A  = [90; 620; 4900];
paramCases(1).B  = [500; 2000; 10000];

% Case 2: BR = 0.98, non-capped beta, homogeneous
% A./B = [0.32; 0.33; 0.33]
paramCases(2).mu = 1;
paramCases(2).A  = [160; 660; 3300];
paramCases(2).B  = [500; 2000; 10000];

% Case 3: BR = 0.80, non-capped beta, heterogeneous
% A./B = [0.14; 0.26; 0.40]
paramCases(3).mu = 1;
paramCases(3).A  = [70; 520; 4000];
paramCases(3).B  = [500; 2000; 10000];

% Case 4: BR = 0.80, non-capped beta, homogeneous
% A./B = [0.26; 0.27; 0.27]
paramCases(4).mu = 1;
paramCases(4).A  = [130; 540; 2700];
paramCases(4).B  = [500; 2000; 10000];

nCases = numel(paramCases);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over parameter cases
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cc = 1:nCases
    
    mu_true = paramCases(cc).mu;
    A_true  = paramCases(cc).A(:);
    B_true  = paramCases(cc).B(:);
    
    % Sort decay coefficients and reorder excitation coefficients
    [B_true,ord] = sort(B_true);
    A_true = A_true(ord);
    
    theta_true = [mu_true; A_true];
    p = K + 1;
    
    BR = sum(A_true ./ B_true);
    
    fprintf('\n=========================================\n');
    fprintf('Case %d / %d\n',cc,nCases);
    fprintf('mu = %.6g\n',mu_true);
    fprintf('A  = %s\n',mat2str(A_true(:)',6));
    fprintf('B  = %s\n',mat2str(B_true(:)',6));
    fprintf('Branching ratio = %.6f\n',BR);
    fprintf('=========================================\n');
    
    if BR >= 1
        warning('Case %d has branching ratio >= 1. Skipping.',cc);
        continue;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Storage
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z_store         = NaN(p,Nsim);
    theta_hat_store = NaN(p,Nsim);
    Delta_store     = NaN(1,Nsim);
    TDelta_store    = NaN(1,Nsim);
    
    status_store  = false(1,Nsim);
    message_store = strings(1,Nsim);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop over the replications
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    parfor ii = 1:Nsim
        
        rng(42 + 100000*cc + ii,'twister');
        
        z_i = NaN(p,1);
        theta_hat_i = NaN(p,1);
        Delta_i = NaN;
        TDelta_i = NaN;
        
        status_i = false;
        message_i = "ok";
        
        try
            % Simulate Hawkes process
            [eventsTimes,~,~] = ...
                simulate_mhawkes(mu_true,A_true,B_true,T);
            
            % Compute standardized OLS statistic using known beta
            out = compute_hawkes_ols_statistic( ...
                eventsTimes,T,c_user,B_true,theta_true, ...
                mT,thetaWeight);
            
            z_i = out.z;
            theta_hat_i = out.theta_hat;
            Delta_i = out.Delta;
            TDelta_i = out.TDelta;
            
            status_i = true;
            
        catch ME
            message_i = string(ME.message);
            status_i = false;
        end
        
        z_store(:,ii) = z_i;
        theta_hat_store(:,ii) = theta_hat_i;
        Delta_store(ii) = Delta_i;
        TDelta_store(ii) = TDelta_i;
        
        status_store(ii) = status_i;
        message_store(ii) = message_i;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Diagnostics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    valid = status_store ...
        & all(isfinite(z_store),1) ...
        & all(isfinite(theta_hat_store),1);
    
    fprintf('Valid replications: %d / %d\n',sum(valid),Nsim);
    
    if any(valid)
        fprintf('Mean standardized statistic:\n');
        disp(mean(z_store(:,valid),2,'omitnan'));
        
        fprintf('Standard deviation of standardized statistic:\n');
        disp(std(z_store(:,valid),0,2,'omitnan'));
        
        fprintf('Mean OLS estimator:\n');
        disp(mean(theta_hat_store(:,valid),2,'omitnan'));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save one MAT file per parameter case
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    outFile = fullfile(results_dir, ...
        sprintf('hawkes_CLT_known_beta_case%d_K%d_T%d_Nsim%d_c%.6g.mat', ...
        cc,K,T,Nsim,c_user));
    
    save(outFile, ...
        'cc','K','T','Nsim', ...
        'c_user','c_grid','grid_all', ...
        'mT','thetaWeight', ...
        'mu_true','A_true','B_true','theta_true','BR', ...
        'z_store','theta_hat_store', ...
        'Delta_store','TDelta_store', ...
        'status_store','message_store','valid', ...
        '-v7.3');
    
    fprintf('Saved case %d results in:\n%s\n',cc,outFile);
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QQ plots of standardized OLS statistics: known beta only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two figures are produced:
%   Figure 1: branching ratio 0.98
%   Figure 2: branching ratio 0.80
%
% In each figure:
%   - 4 rows: z_1, z_2, z_3, z_4
%   - left column: homogeneous case
%   - right column: heterogeneous case
%
% Parameter-case ordering:
%   Case 1: BR = 0.98, heterogeneous
%   Case 2: BR = 0.98, homogeneous
%   Case 3: BR = 0.80, heterogeneous
%   Case 4: BR = 0.80, homogeneous
%%

clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paths and simulation settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results_dir = fullfile(pwd,'results_dist');

K = 3; T = 10000; Nsim = 2000;

grid_all = logspace(-6.5,2,120).';
c_user = grid_all(60);

componentNames = {
    '$z_1$'
    '$z_2$'
    '$z_3$'
    '$z_4$'
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure definitions
% Left column: homogeneous
% Right column: heterogeneous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figureCases = struct([]);

figureCases(1).BRlabel = '0.98';
figureCases(1).homogeneousCase = 2;
figureCases(1).heterogeneousCase = 1;

figureCases(2).BRlabel = '0.80';
figureCases(2).homogeneousCase = 4;
figureCases(2).heterogeneousCase = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over Branching Ratio settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ff = 1:numel(figureCases)
    
    caseHom = figureCases(ff).homogeneousCase;
    caseHet = figureCases(ff).heterogeneousCase;
    BRlabel = figureCases(ff).BRlabel;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load homogeneous case
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    matFileHom = fullfile(results_dir, ...
        sprintf('hawkes_CLT_known_beta_case%d_K%d_T%d_Nsim%d_c%.6g.mat', ...
        caseHom,K,T,Nsim,c_user));
    
    if ~isfile(matFileHom)
        warning('Homogeneous-case file not found:\n%s',matFileHom);
        continue;
    end
    
    SHom = load(matFileHom, ...
        'z_store','status_store', ...
        'mu_true','A_true','B_true','BR');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load heterogeneous case
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    matFileHet = fullfile(results_dir, ...
        sprintf('hawkes_CLT_known_beta_case%d_K%d_T%d_Nsim%d_c%.6g.mat', ...
        caseHet,K,T,Nsim,c_user));
    
    if ~isfile(matFileHet)
        warning('Heterogeneous-case file not found:\n%s',matFileHet);
        continue;
    end
    
    SHet = load(matFileHet, ...
        'z_store','status_store', ...
        'mu_true','A_true','B_true','BR');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create figure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure( ...
        'Visible','on', ...
        'Units','inches', ...
        'Position',[1 1 10 14]);
    
    tiledlayout(4,2, ...
        'TileSpacing','compact', ...
        'Padding','compact');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % QQ plots
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for jj = 1:(K+1)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Left panel: homogeneous case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        axHom = nexttile;
        
        validHom = ...
            SHom.status_store(:).' ...
            & isfinite(SHom.z_store(jj,:));
        
        zHom = SHom.z_store(jj,validHom);
        zHom = zHom(:);
        
        if numel(zHom) >= 3
            qqplot(zHom);
            grid on;
        else
            axis off;
            text(0.5,0.5,'Insufficient valid observations', ...
                'HorizontalAlignment','center');
        end
        
        title(componentNames{jj}, ...
            'Interpreter','latex');
        
        xlabel('Theoretical quantiles');
        ylabel('Sample quantiles');
        
        if jj == 1
            text(axHom,0.5,1.18,'Homogeneous', ...
                'Units','normalized', ...
                'HorizontalAlignment','center', ...
                'FontWeight','bold', ...
                'FontSize',12);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Right panel: heterogeneous case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        axHet = nexttile;
        
        validHet = ...
            SHet.status_store(:).' ...
            & isfinite(SHet.z_store(jj,:));
        
        zHet = SHet.z_store(jj,validHet);
        zHet = zHet(:);
        
        if numel(zHet) >= 3
            qqplot(zHet);
            grid on;
        else
            axis off;
            text(0.5,0.5,'Insufficient valid observations', ...
                'HorizontalAlignment','center');
        end
        
        title(componentNames{jj}, ...
            'Interpreter','latex');
        
        xlabel('Theoretical quantiles');
        ylabel('Sample quantiles');
        
        if jj == 1
            text(axHet,0.5,1.18,'Heterogeneous', ...
                'Units','normalized', ...
                'HorizontalAlignment','center', ...
                'FontWeight','bold', ...
                'FontSize',12);
        end
    end
    
    sgtitle(sprintf('Branching ratio $= %s$',BRlabel), ...
        'Interpreter','latex', ...
        'FontSize',14);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save figure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    BRtag = strrep(BRlabel,'.','_');
    
    pdfFile = fullfile(results_dir, ...
        sprintf(['QQplots_known_beta_BR%s_' ...
        'K%d_T%d_Nsim%d_c%.6g.pdf'], ...
        BRtag,K,T,Nsim,c_user));
    
    exportgraphics(fig,pdfFile,'ContentType','vector');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Display diagnostics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    validCountsHom = sum( ...
        isfinite(SHom.z_store) ...
        & repmat(SHom.status_store(:).',K+1,1), ...
        2);
    
    validCountsHet = sum( ...
        isfinite(SHet.z_store) ...
        & repmat(SHet.status_store(:).',K+1,1), ...
        2);
    
    fprintf('\n============================================\n');
    fprintf('Branching ratio: %s\n',BRlabel);
    fprintf('Homogeneous parameter case:   %d\n',caseHom);
    fprintf('Heterogeneous parameter case: %d\n',caseHet);
    
    fprintf('Homogeneous valid counts:   %s\n', ...
        mat2str(validCountsHom.'));
    
    fprintf('Heterogeneous valid counts: %s\n', ...
        mat2str(validCountsHet.'));
    
    fprintf('Saved figure:\n%s\n',pdfFile);
end