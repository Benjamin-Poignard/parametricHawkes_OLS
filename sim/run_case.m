function run_case(Sim,T,grid_val,results_dir,BR_label,targetBR,caseID,mu_value,doIntegerA,alphaMin,betaMin)

% Code for simulation experiments for l2 error
rng(42,'twister');

fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
fprintf('Running section %s\n',caseID);
fprintf('BR label       = %s\n',BR_label);
fprintf('Target BR      = %.4f\n',targetBR);
fprintf('Minimum alpha  = %g\n',alphaMin);
fprintf('Minimum beta   = %g\n',betaMin);
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

cases = build_cases( ...
    caseID,mu_value,targetBR,doIntegerA,alphaMin,betaMin);

for cc = 1:numel(cases)
    
    mu = cases{cc}.mu;
    A  = cases{cc}.A(:);
    B  = cases{cc}.B(:);
    
    n = numel(A);
    typeStr = cases{cc}.type;
    actualBR = sum(A./B);
    
    fprintf('\n====================================================\n');
    fprintf('Running case %d / %d\n',cc,numel(cases));
    fprintf('Section       = %s\n',caseID);
    fprintf('T             = %g\n',T);
    fprintf('mu            = %g\n',mu);
    fprintf('n             = %d\n',n);
    fprintf('type          = %s\n',typeStr);
    fprintf('A             = [%s]\n',num2str(A.'));
    fprintf('B             = [%s]\n',num2str(B.'));
    fprintf('min(A)        = %.10g\n',min(A));
    fprintf('min(B)        = %.10g\n',min(B));
    fprintf('alpha/beta    = [%s]\n',num2str((A./B).'));
    fprintf('target BR     = %.10f\n',targetBR);
    fprintf('actual BR     = %.10f\n',actualBR);
    fprintf('====================================================\n');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parameter checks
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if actualBR < targetBR - 1e-12 || actualBR >= 1
        error(['Invalid branching ratio: target = %.12g, ' ...
            'actual = %.12g.'],targetBR,actualBR);
    end
    
    if min(A) <= 1
        error('True alpha condition violated: min(A) = %.12g.',min(A));
    end
    
    if min(B) <= 2
        error('True beta condition violated: min(B) = %.12g.',min(B));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Bin size: Delta = c / beta_max
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dtGrid = grid_val./max(B);
    
    nGrid = numel(grid_val);
    
    dist_ols = NaN(nGrid,Sim);
    dist_mle = NaN(Sim,1);
    
    param_true = [mu;A];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Replication loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ll = 1:Sim
        
        fprintf( ...
            'Section %s | case %d/%d | simulation %d/%d\n', ...
            caseID,cc,numel(cases),ll,Sim);
        
        % Simulate Hawkes process
        [eventsTimes,~,~] = simulate_mhawkes(mu,A,B,T);
        
        eventsTimes = eventsTimes(:);
        eventsMarks = ones(size(eventsTimes));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MLE with known beta
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        opts.Display     = 'off';
        opts.MaxIter     = 100000;
        opts.MaxFunEvals = 20e5;
        
        options = optimoptions('fmincon', ...
            'Algorithm','interior-point', ...
            'Display',opts.Display, ...
            'MaxIterations',opts.MaxIter, ...
            'MaxFunctionEvaluations',opts.MaxFunEvals);
        
        theta_init = [ ...
            -0.1 + 1.5*rand(1) + mu; ...
            (-5 + 10*rand(n,1)) + A];
        
        negativeIndex = theta_init < 0;
        
        if any(negativeIndex)
            theta_init(negativeIndex) = ...
                rand(sum(negativeIndex),1);
        end
        
        obj = @(x) HawkesMLE( ...
            x,eventsTimes,eventsMarks,B,T);
        
        theta_hat = fmincon( ...
            obj,theta_init, ...
            [],[],[],[],[],[], ...
            @(x) constr_hawkes(x,B), ...
            options);
        
        theta_hat = theta_hat(:);
        
        dist_mle(ll) = norm(param_true-theta_hat);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % OLS estimation across the grid
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parfor kk = 1:nGrid
            
            variable = bin_events( ...
                eventsTimes,T,dtGrid(kk));
            
            est = hawkes_OLS( ...
                variable.nzIdx, ...
                variable.nzVal, ...
                variable.T, ...
                dtGrid(kk), ...
                B);
            
            theta_ols = est.theta(:);
            
            dist_ols(kk,ll) = ...
                norm(param_true-theta_ols);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Aggregate the errors
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Results = mean(dist_ols,2,'omitnan');
    results_mle = mean(dist_mle,'omitnan');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % File names
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Astr_underscore = strjoin(string(A(:).'),'-');
    Bstr_underscore = strjoin(string(B(:).'),'-');
    
    baseName = sprintf( ...
        '%s_%s_T=%g_mu=%g_n=%d_type=%s_A=%s_B=%s', ...
        caseID, ...
        BR_label, ...
        T, ...
        mu, ...
        n, ...
        matlab.lang.makeValidName(typeStr), ...
        Astr_underscore, ...
        Bstr_underscore);
    
    txtFile = fullfile( ...
        results_dir, ...
        sprintf('results_%s.txt',baseName));
    
    matFile = fullfile( ...
        results_dir, ...
        sprintf('results_%s.mat',baseName));
    
    pdfFile = fullfile( ...
        results_dir, ...
        sprintf('figure_%s.pdf',baseName));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save TXT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fid = fopen(txtFile,'w');
    
    if fid == -1
        error('Cannot open file for writing: %s',txtFile);
    end
    
    fprintf(fid,'====================================================\n');
    fprintf(fid,'Section = %s\n',caseID);
    fprintf(fid,'BR label = %s\n',BR_label);
    fprintf(fid,'T = %g\n',T);
    fprintf(fid,'Sim = %d\n',Sim);
    fprintf(fid,'mu = %g\n',mu);
    fprintf(fid,'n = %d\n',n);
    fprintf(fid,'type = %s\n',typeStr);
    fprintf(fid,'A = [%s]\n',num2str(A.'));
    fprintf(fid,'B = [%s]\n',num2str(B.'));
    fprintf(fid,'min(A) = %.12g\n',min(A));
    fprintf(fid,'min(B) = %.12g\n',min(B));
    fprintf(fid,'alpha/beta = [%s]\n',num2str((A./B).'));
    fprintf(fid,'target branching = %.12g\n',targetBR);
    fprintf(fid,'actual branching = %.12g\n',actualBR);
    fprintf(fid,'====================================================\n\n');
    
    fprintf(fid,'grid\tdt_beta_max\tOLS_beta_max\n');
    
    for kk = 1:nGrid
        fprintf(fid,'%.12g\t%.12g\t%.12g\n', ...
            grid_val(kk), ...
            dtGrid(kk), ...
            Results(kk));
    end
    
    fprintf(fid,'\nMLE = %.12g\n',results_mle);
    
    fclose(fid);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save MAT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    save(matFile, ...
        'caseID', ...
        'BR_label', ...
        'targetBR', ...
        'actualBR', ...
        'T', ...
        'Sim', ...
        'mu', ...
        'n', ...
        'typeStr', ...
        'A', ...
        'B', ...
        'alphaMin', ...
        'betaMin', ...
        'param_true', ...
        'grid_val', ...
        'dtGrid', ...
        'dist_ols', ...
        'dist_mle', ...
        'Results', ...
        'results_mle', ...
        '-v7.3');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plots
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Visible','off');
    
    semilogx( ...
        grid_val,Results, ...
        'r','LineWidth',1.5);
    
    hold on;
    
    semilogx( ...
        grid_val, ...
        results_mle*ones(size(grid_val)), ...
        'k','LineWidth',1.5);
    
    finiteValues = [ ...
        Results(isfinite(Results)); ...
        results_mle];
    
    finiteValues = finiteValues(isfinite(finiteValues));
    
    if isempty(finiteValues)
        yMax = 1;
    else
        yMax = max(finiteValues);
        
        if yMax <= 0
            yMax = 1;
        end
    end
    
    ylim([0,1.15*yMax]);
    
    grid on;
    
    xlabel( ...
        'Grid parameter $c$', ...
        'Interpreter','latex');
    
    ylabel( ...
        '$\ell_2$ error', ...
        'Interpreter','latex');
    
    legend( ...
        { ...
        'OLS, $\Delta=c/\beta_{\max}$', ...
        'MLE' ...
        }, ...
        'Location','best', ...
        'Interpreter','latex', ...
        'FontSize',12);
    
    width = 9;
    height = 5;
    
    set(fig,'Units','inches');
    set(fig,'Position',[1,1,width,height]);
    set(fig,'PaperPositionMode','auto');
    
    exportgraphics( ...
        fig,pdfFile, ...
        'ContentType','vector');
    
    close(fig);
    
    fprintf('Saved:\n');
    fprintf('  %s\n',txtFile);
    fprintf('  %s\n',matFile);
    fprintf('  %s\n',pdfFile);
end