%% Rolling-Window Calibration and Out-of-Sample RMSE Evaluation (SLTM and Vasicek)
clear; clc; close all;

% === Load Data ===
filename = 'daily-treasury-rates.csv';
tbl = readtable(filename, 'VariableNamingRule', 'preserve');
tbl(:, "1.5 Month") = [];  % remove column if present

rates = tbl{:, 2:end} / 100;  % convert to decimals
N_total = size(rates, 1);

T = [1, 2, 3, 4, 6]/12;           % short-term in years
T = [T, 1, 2, 3, 5, 7, 10, 20, 30]';
rates = rates(:, 1:end-2);       % remove 20y and 30y
T = T(1:end-2);                  % match maturities

window_size = 10;
N = N_total - window_size;

params_all_stoch = NaN(N,6);
params_all_vas = NaN(N,4);
rmse_oos_stoch = NaN(N,1);
rmse_oos_vas = NaN(N,1);
rmses_in_sample_stoch = NaN(N,1);
rmses_in_sample_vas = NaN(N,1);

x0_stoch = [0.03, 0.04, 0.8, 0.05, 0.5, 0.06];
x0_vas = [0.01, 0.03, 0.1, 0.01];

lb_stoch = [0, 0, 0.01, 0.0001, 0.01, 0];
ub_stoch = [0.5, 0.5, 5, 0.1, 5, 0.5];
lb_vas = [0, 0, 0.01, 0.0001];
ub_vas = [0.5, 5, 5, 0.5];

opts = optimset('Display','off', 'TolFun',1e-8, 'TolX',1e-8);

for i = 1:N
    idx = i:(i+window_size-1);
    curves = rates(idx, :);

    % --- Stochastic Model ---
    objective_stoch = @(params) sum(arrayfun(@(j) norm(...
        -log(arrayfun(@(t) safe_exp_characteristic(params, t), T)) ./ T - curves(j,:)', 1), 1:window_size));

    [params_s, fval_s] = fmincon(objective_stoch, x0_stoch, [], [], [], [], lb_stoch, ub_stoch, [], opts);
    x0_stoch = params_s;  % update initial guess
    params_all_stoch(i,:) = params_s;
    rmses_in_sample_stoch(i) = fval_s / window_size;

    future_curve = rates(i+window_size, :);
    pred_prices = arrayfun(@(t) safe_exp_characteristic(params_s, t), T);
    pred_yields = -log(pred_prices) ./ T;
    rmse_oos_stoch(i) = norm(pred_yields - future_curve', 2);

    % --- Vasicek Model ---
    vas_price = @(params, Tvec) exp(-(params(2) + (params(1)-params(2))*(1 - exp(-params(3)*Tvec)) ./ (params(3)*Tvec) ...
        - 0.5 * params(4)^2 / params(3)^2 * (1 - exp(-params(3)*Tvec)).^2 ./ Tvec));
    objective_vas = @(params) sum(arrayfun(@(j) norm(...
        -log(vas_price(params, T)) ./ T - curves(j,:)', 1), 1:window_size));

    [params_v, fval_v] = fmincon(objective_vas, x0_vas, [], [], [], [], lb_vas, ub_vas, [], opts);
    x0_vas = params_v;  % update initial guess
    params_all_vas(i,:) = params_v;
    rmses_in_sample_vas(i) = fval_v / window_size;

    pred_prices_vas = vas_price(params_v, T);
    pred_yields_vas = -log(pred_prices_vas) ./ T;
    rmse_oos_vas(i) = norm(pred_yields_vas - future_curve', 2);
end

% === Plotting ===
time_axis = (1:N) + window_size;
param_names_stoch = {'r0','theta0','kappa','s','v','m'};
param_names_vas = {'r0','theta','kappa','sigma'};

figure('Name','Evolução dos Parâmetros – SLTM','NumberTitle','off');
for j = 1:6
    subplot(3,2,j);
    plot(time_axis, params_all_stoch(:,j), 'LineWidth', 1.2);
    title(['Parâmetro ', param_names_stoch{j}]);
    xlabel('Tempo'); ylabel(param_names_stoch{j});
    grid on;
end

figure('Name','Evolução dos Parâmetros – Vasicek','NumberTitle','off');
for j = 1:4
    subplot(2,2,j);
    plot(time_axis, params_all_vas(:,j), 'LineWidth', 1.2);
    title(['Parâmetro ', param_names_vas{j}]);
    xlabel('Tempo'); ylabel(param_names_vas{j});
    grid on;
end

figure('Name','Out-of-Sample RMSE','NumberTitle','off');
plot(time_axis, rmse_oos_stoch, 'r-', 'LineWidth', 1.5); hold on;
plot(time_axis, rmse_oos_vas, 'b--', 'LineWidth', 1.5);
xlabel('Time (days)'); ylabel('RMSE');
title('Out-of-Sample RMSE – SLTM vs Vasicek');
legend('SLTM','Vasicek'); grid on;

mean(rmse_oos_stoch)
std(rmse_oos_stoch)
mean(rmse_oos_vas)
std(rmse_oos_vas)
%% --- Funções Auxiliares ---
function val = safe_exp_characteristic(params, t)
    try
        r0 = params(1); theta0 = params(2); kappa = params(3);
        s = params(4); v = params(5); m = params(6);

        tol = 1e-4;
        if abs(kappa - v) < tol
            phi = compute_characteristic_function_limit(1i, t, r0, theta0, v, s, m);
        else
            phi = compute_characteristic_function2(1i, t, r0, theta0, kappa, s, v, m);
        end
        val = real(phi); if ~isfinite(val) || val <= 0, val = NaN; end
    catch
        val = NaN;
    end
end

function phi_RT = compute_characteristic_function_limit(u, T, r0, theta0, k, s, m)
    q = 1i * u; exp_vDelta = exp(k * T);
    phi = (q * T * exp(-2 * k * T) / 2) * (-s^2 * q * T + 2 * (m * k + r0 + theta0) * exp_vDelta);
    phi_RT = exp(phi); if ~isfinite(phi_RT) || real(phi_RT) <= 0, phi_RT = NaN; end
end

function phi_RT = compute_characteristic_function2(u, t, r0, theta0, k, s, v, m)
    q = 1i * u;
    beta1 = (q - q * exp(-k * t)) / k;
    beta2 = (q * (k - v + v * exp(-k * t))) / (v * (k - v)) - (k * q * exp(-v * t)) / (v * (k - v));
    term1 = (k * m * q * t) / (k - v) - (m * q * v * t) / (k - v);
    term2 = - (m * q * v * (exp(-k * t) - 1)) / (k * (k - v));
    term3 = (k * m * q * (exp(-v * t) - 1)) / (v * (k - v));
    q2 = q^2;
    num_s = ...
        (q2 * (3 * k^4 - 4 * k^4 * exp(-v * t) + k^4 * exp(-2 * v * t))) / 2 + ...
        (q2 * v^3 * (k * exp(-2 * k * t) - k + 2 * k^2 * t)) / 2 - ...
        (q2 * v * (2 * k^4 * t + k^3 - k^3 * exp(-2 * v * t))) / 2 + ...
        (q2 * v^2 * (2 * k^3 * t - 4 * k^2 * exp(-k * t - v * t) - 4 * k^2 + 4 * k^2 * exp(-k * t) + 4 * k^2 * exp(-v * t))) / 2 - ...
        (q2 * v^4 * (4 * exp(-k * t) - exp(-2 * k * t) + 2 * k * t - 3)) / 2;
    denom_s = 2 * k * v^3 * (k + v) * (k - v)^2;
    alpha = term1 + term2 + term3 - (s^2 * num_s) / denom_s;
    phi = alpha + beta1 * r0 + beta2 * theta0;
    if abs(real(phi)) > 700, phi_RT = NaN; return; end
    phi_RT = exp(phi); if ~isfinite(phi_RT) || real(phi_RT) <= 0, phi_RT = NaN; end
end