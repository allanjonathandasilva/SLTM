%% Calibration of interest rate models using Treasury .csv data
clear; clc; close all;

% === Reading the CSV file ===
filename = 'daily-treasury-rates.csv';
tbl = readtable(filename, 'VariableNamingRule', 'preserve');

% Remove the "1.5 Month" column
tbl(:, "1.5 Month") = [];

% Extract yields and convert to decimal
rates = tbl{:, 2:end} / 100;  % all rate columns (excludes the Date column)
N = size(rates, 1);

% Maturities in years (13 maturities after removing "1.5 Month")
T = [1, 2, 3, 4, 6] / 12;           % months converted to years
T = [T, 1, 2, 3, 5, 7, 10, 20, 30]'; % years

rates = rates(:, 1:end-2);  % remove the last 2 columns (20 Yr and 30 Yr)
T = T(1:end-2);             % remove the last two maturities

% Vectors for errors and parameters
rmse_vas = zeros(N,1);
rmse_stoch = zeros(N,1);
params_vas_all = NaN(N,4);
params_stoch_all = NaN(N,6);

% Optimization settings
opts = optimset('Display','off', 'TolFun',1e-12, 'TolX',1e-12, 'MaxIter',2500);

% Initial guesses
x0_vas = [0.1, 0.1, 0.5, 0.01];
x0_stoch = [0.03, 0.04, 0.4, 0.02, 0.25, 0.05];

lb_vas = [0, 0, 0.01, 0.0001];  ub_vas = [5, 10, 5, 0.5];
lb_stoch = [0, 0, 0.01, 0.0001, 0.01, 0];  ub_stoch = [0.5, 0.5, 5, 0.25, 5, 0.5];

% Loop over dates
for i = 1:N
    yields = rates(i, :);
    if any(isnan(yields))
        rmse_vas(i) = NaN;
        rmse_stoch(i) = NaN;
        continue;
    end

    %% Vasicek
    vasicek_price = @(params, T) ...
        exp(-(params(2) + (params(1) - params(2)) * (1 - exp(-params(3)*T)) ./ (params(3)*T) ...
            - 0.5 * params(4)^2 / params(3)^2 * (1 - exp(-params(3)*T)).^2 ./ T));

    objective_vas = @(params) ...
        norm(-log(vasicek_price(params, T)) ./ T - yields', 1);

    [params_vas, fval_vas] = fmincon(objective_vas, x0_vas, [], [], [], [], lb_vas, ub_vas, [], opts);
    rmse_vas(i) = fval_vas;
    params_vas_all(i,:) = params_vas;  % store
    x0_vas = params_vas;               % update guess for the next date

    %% Stochastic model
    stoch_mean_price = @(params, Tvec) ...
        arrayfun(@(t) safe_exp_characteristic(params, t), Tvec);

    objective_stoch = @(params) ...
        norm(-log(stoch_mean_price(params, T)) ./ T - yields', 1);

% Diagnostic test for initial guess x0_stoch
% disp('--- Diagnostic test for initial guess x0_stoch ---');
% disp('Parameters:'); disp(x0_stoch);
% 
% f = stoch_mean_price(x0_stoch, T);
% disp('Price function (stoch_mean_price):'); disp(f);
% 
% if any(isnan(f)) || any(~isfinite(f)) || any(f <= 0)
%     error('>>> OBJECTIVE FUNCTION HAS INVALID VALUE AT x0_stoch <<<');
% end

    [params_stoch, fval_stoch] = fmincon(objective_stoch, x0_stoch, [], [], [], [], lb_stoch, ub_stoch, [], opts);
    rmse_stoch(i) = fval_stoch;
    params_stoch_all(i,:) = params_stoch;
    x0_stoch = params_stoch;  % update guess

    %% Plot of fitted curve
    if(i>1)
        curve_vas = -log(vasicek_price(params_vas, T)) ./ T;
        curve_stoch = -log(stoch_mean_price(params_stoch, T)) ./ T;
    
        figure('Name', sprintf('Fitted Curve - Date %d', i), 'NumberTitle','off');
        plot(T, yields, 'ko-', 'LineWidth', 1.5); hold on;
        plot(T, curve_vas, 'b--', 'LineWidth', 1.5);
        plot(T, curve_stoch, 'r-.', 'LineWidth', 1.5);
        xlabel('Maturity (years)');
        ylabel('Yield (decimal)');
        title(sprintf('Date %d – Observed vs Vasicek vs Stochastic', i));
        legend('Observed', 'Vasicek', 'Stochastic Mean', 'Location', 'best');
        grid on;
        drawnow;
    end
end

%%
valid_idx = 2:N;
% RMSEs
rmse_vas  = rmse_vas(valid_idx);
rmse_stoch = rmse_stoch(valid_idx);

% Parameters
params_vas   = params_vas_all(valid_idx, :);
params_stoch = params_stoch_all(valid_idx, :);

%% Final Statistics
fprintf('\n=== Calibration Statistics ===\n');
fprintf('Vasicek Model\n');
fprintf('  Mean RMSE  : %.6f\n', mean(rmse_vas, 'omitnan'));
fprintf('  Std. Dev. RMSE : %.6f\n', std(rmse_vas, 'omitnan'));
fprintf('  Max Error  : %.6f\n', max(rmse_vas));

fprintf('\nStochastic Mean Model\n');
fprintf('  Mean RMSE  : %.6f\n', mean(rmse_stoch, 'omitnan'));
fprintf('  Std. Dev. RMSE : %.6f\n', std(rmse_stoch, 'omitnan'));
fprintf('  Max Error  : %.6f\n', max(rmse_stoch));

%% Display of Estimated Parameters
fprintf('\n=== Estimated Parameters – Vasicek Model ===\n');
headers_vas = {'r0','theta','kappa','sigma'};
disp(array2table(params_vas, 'VariableNames', headers_vas));

fprintf('\n=== Estimated Parameters – Stochastic Mean Model ===\n');
headers_stoch = {'r0','theta0','kappa','s','v','m'};
disp(array2table(params_stoch, 'VariableNames', headers_stoch));

%% === Descriptive Statistics and Correlations for Stochastic Parameters ===

% Ignore rows with NaNs
valid_idx = all(isfinite(params_stoch_all), 2);
params_valid = params_stoch_all(valid_idx, :);

% Descriptive statistics
fprintf('\n=== Descriptive Statistics – Stochastic Mean Model ===\n');
headers = {'r0','theta0','kappa','s','v','m'};
for j = 1:length(headers)
    col = params_valid(:, j);
    fprintf('%s: Mean = %.5f, Std = %.5f, Min = %.5f, Max = %.5f\n', ...
        headers{j}, mean(col), std(col), min(col), max(col));
end

% Correlation matrix
fprintf('\n=== Correlation Matrix – Stochastic Parameters ===\n');
corr_matrix = corr(params_valid, 'Rows', 'complete');
disp(array2table(corr_matrix, 'VariableNames', headers, 'RowNames', headers));

% (Optional) Histograms
figure('Name', 'Histograms of Stochastic Parameters', 'NumberTitle', 'off');
for j = 1:length(headers)
    subplot(2, 3, j);
    histogram(params_valid(:, j), 20, 'FaceColor', [0.2 0.4 0.6]);
    title(headers{j}); xlabel('Value'); ylabel('Frequency');
end

%% === Descriptive Statistics and Correlations for Vasicek Parameters ===

% Ignore rows with NaNs
valid_idx_vas = all(isfinite(params_vas_all), 2);
params_vas_valid = params_vas_all(valid_idx_vas, :);

% Descriptive statistics
fprintf('\n=== Descriptive Statistics – Vasicek Model ===\n');
headers_vas = {'r0','theta','kappa','sigma'};
for j = 1:length(headers_vas)
    col = params_vas_valid(:, j);
    fprintf('%s: Mean = %.5f, Std = %.5f, Min = %.5f, Max = %.5f\n', ...
        headers_vas{j}, mean(col), std(col), min(col), max(col));
end

% Correlation matrix
fprintf('\n=== Correlation Matrix – Vasicek Parameters ===\n');
corr_matrix_vas = corr(params_vas_valid, 'Rows', 'complete');
disp(array2table(corr_matrix_vas, 'VariableNames', headers_vas, 'RowNames', headers_vas));

% (Optional) Histograms
figure('Name', 'Histograms of Vasicek Parameters', 'NumberTitle', 'off');
for j = 1:length(headers_vas)
    subplot(2, 2, j);
    histogram(params_vas_valid(:, j), 20, 'FaceColor', [0.5 0.3 0.2]);
    title(headers_vas{j}); xlabel('Value'); ylabel('Frequency');
end

%% === Export LaTeX Tables Formatted with \\ and & Using 5 Decimal Places ===

% Output files
fileID1 = fopen('vasicek_parameters_table.tex', 'w');
fileID2 = fopen('stoch_parameters_table.tex', 'w');

% Format: 4 columns with 5 decimal places
for i = 1:size(params_vas,1)
    fprintf(fileID1, '%.5f & %.5f & %.5f & %.5f \\\\\n', params_vas(i,1), ...
            params_vas(i,2), params_vas(i,3), params_vas(i,4));
end

% Format: 6 columns with 5 decimal places
for i = 1:size(params_stoch,1)
    fprintf(fileID2, '%.5f & %.5f & %.5f & %.5f & %.5f & %.5f \\\\\n', ...
        params_stoch(i,1), params_stoch(i,2), params_stoch(i,3), ...
        params_stoch(i,4), params_stoch(i,5), params_stoch(i,6));
end

% Close files
fclose(fileID1);
fclose(fileID2);


%%

function val = safe_exp_characteristic(params, t)
    % safe_exp_characteristic — Evaluates the price of a zero-coupon bond under the
    % stochastic mean model, with robustness checks and handling of the degenerate case k ≈ v.

    try
        % Extract parameters from the input vector
        r0     = params(1);  % Initial short rate
        theta0 = params(2);  % Initial long-term mean
        kappa  = params(3);  % Mean reversion speed of r(t)
        s      = params(4);  % Volatility of theta(t)
        v      = params(5);  % Mean reversion speed of theta(t)
        m      = params(6);  % Long-run mean of theta(t)

        % Basic parameter validity checks
        if any(~isfinite(params)) || kappa <= 0 || v <= 0 || s <= 0
            val = NaN;
            return;
        end

        % Tolerance threshold to treat kappa approximately equal to v
        tol = 1e-4;

        % Select the appropriate version of the characteristic function
        if abs(kappa - v) < tol
            phi = compute_characteristic_function_limit(1i, t, r0, theta0, v, s, m);
        else
            phi = compute_characteristic_function2(1i, t, r0, theta0, kappa, s, v, m);
        end

        % Post-calculation validity checks
        if ~isfinite(phi) || real(phi) <= 0
            val = NaN;
            return;
        end

        % Bond price derived from the characteristic function (real part)
        val = real(phi);

    catch
        % In case of any unexpected error, return NaN
        val = NaN;
    end
end

%%
function phi_RT = compute_characteristic_function2(u, t, r0, theta0, k, s, v, m)
    % Closed-form characteristic function for R_T = ∫₀^t r(s) ds
    % Based on the provided analytical formulas
    % u: evaluation point (typically real positive for Laplace or imaginary for Fourier transform)

    q = 1i * u;  % Characteristic function argument (iu)

    % Coefficient beta_1
    beta1 = (q - q * exp(-k * t)) / k;

    % Coefficient beta_2
    beta2 = (q * (k - v + v * exp(-k * t))) / (v * (k - v)) ...
            - (k * q * exp(-t * v)) / (v * (k - v));

    % Coefficient alpha (symbolically reorganized formula)
    term1 = (k * m * q * t) / (k - v) - (m * q * v * t) / (k - v);
    term2 = - (m * q * v * (exp(-k * t) - 1)) / (k * (k - v));
    term3 = (k * m * q * (exp(-v * t) - 1)) / (v * (k - v));

    % Complex term involving s^2 * q^2 * (...)
    q2 = q^2;
    num_s = ...
        (q2 * (3 * k^4 - 4 * k^4 * exp(-v * t) + k^4 * exp(-2 * v * t))) / 2 ...
      + (q2 * v^3 * (k * exp(-2 * k * t) - k + 2 * k^2 * t)) / 2 ...
      - (q2 * v * (2 * k^4 * t + k^3 - k^3 * exp(-2 * v * t))) / 2 ...
      + (q2 * v^2 * (2 * k^3 * t - 4 * k^2 * exp(-k * t - v * t) ...
         - 4 * k^2 + 4 * k^2 * exp(-k * t) + 4 * k^2 * exp(-v * t))) / 2 ...
      - (q2 * v^4 * (4 * exp(-k * t) - exp(-2 * k * t) + 2 * k * t - 3)) / 2;

    denom_s = 2 * k * v^3 * (k + v) * (k - v)^2;

    alpha = term1 + term2 + term3 - (s^2 * num_s) / denom_s;

    % Characteristic function expression
    phi = alpha + beta1 * r0 + beta2 * theta0;

    % Overflow protection
    if abs(real(phi)) > 700
        phi_RT = NaN; return;
    end

    phi_RT = exp(phi);

    % Final numerical stability check
    if ~isfinite(phi_RT) || real(phi_RT) <= 0
        phi_RT = NaN;
    end
end

