clc
clear all
format long
tic

%% Model with Stochastic Long-Term Mean (theta(t) is stochastic)
r0      = 0.149;     % Initial short rate
theta0  = 0.022;     % Initial value of the long-term mean
m       = 0.1213;    % Long-run average of the long-term mean
s       = 0.01;      % Volatility of theta(t)
k       = 1.2392;    % Mean reversion speed of r(t)
v       = 1.2375;    % Mean reversion speed of theta(t)

% Maturities for which duration and convexity will be computed
mkttimes = [1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9]';

%% Parameters for the classical Vasicek model
r0v     = 0.10237;
kappav  = 0.0008;
sv      = 0.00000005;
thetav  = 3.69;

%% Initialization of output arrays
h = 1e-4; % Finite difference increment
Duration_vas     = zeros(length(mkttimes), 1);
Convexity_vas    = zeros(length(mkttimes), 1);
Duration_stoch   = zeros(length(mkttimes), 1);
Convexity_stoch  = zeros(length(mkttimes), 1);

% Loop over maturities
for i = 1:length(mkttimes)
    T = mkttimes(i);
    
    %% Classical Vasicek Model
    phi_RT       = characteristic_function_vasicek(1i, T, r0v, kappav, sv, thetav);
    phi_RT_plus  = characteristic_function_vasicek(1i, T, r0v + h, kappav, sv, thetav);
    phi_RT_minus = characteristic_function_vasicek(1i, T, r0v - h, kappav, sv, thetav);
    
    P        = phi_RT;
    P_plus   = phi_RT_plus;
    P_minus  = phi_RT_minus;
    
    Duration_vas(i)   = -(P_plus - P_minus) / (2 * h * P);
    Convexity_vas(i)  = (P_plus - 2 * P + P_minus) / (h^2 * P);

    %% Model with Stochastic Long-Term Mean
    phi_stoch       = compute_characteristic_function2(1i, T, r0, theta0, k, s, v, m);
    phi_stoch_plus  = compute_characteristic_function2(1i, T, r0 + h, theta0, k, s, v, m);
    phi_stoch_minus = compute_characteristic_function2(1i, T, r0 - h, theta0, k, s, v, m);
    
    P2        = phi_stoch;
    P2_plus   = phi_stoch_plus;
    P2_minus  = phi_stoch_minus;
    
    Duration_stoch(i)   = -(P2_plus - P2_minus) / (2 * h * P2);
    Convexity_stoch(i)  = (P2_plus - 2 * P2 + P2_minus) / (h^2 * P2);
end

%% Plotting duration and convexity
figure;
plot(mkttimes, Duration_stoch, 'r-*', mkttimes, Duration_vas, 'b-o');
xlabel('Maturity (years)'); ylabel('Duration'); grid on;
legend('Stochastic long-term mean', 'Vasicek'); title('Duration Comparison');

figure;
plot(mkttimes, Convexity_stoch, 'r-*', mkttimes, Convexity_vas, 'b-o');
xlabel('Maturity (years)'); ylabel('Convexity'); grid on;
legend('Stochastic long-term mean', 'Vasicek'); title('Convexity Comparison');

%% Print the comparative table: Duration and Convexity
fprintf('\nComparative Table: Duration and Convexity\n');
fprintf('%10s | %20s %20s | %20s %20s\n', 'Maturity', ...
        'Duration (Stochastic)', 'Duration (Vasicek)', ...
        'Convexity (Stochastic)', 'Convexity (Vasicek)');

for i = 1:length(mkttimes)
    fprintf('%10.2f | %20.6f %20.6f | %20.6f %20.6f\n', mkttimes(i), ...
        Duration_stoch(i), Duration_vas(i), Convexity_stoch(i), Convexity_vas(i));
end

%% Compute the sensitivity of the option price with respect to theta0
ThetaSensitivity_stoch = zeros(length(mkttimes), 1);

for i = 1:length(mkttimes)
    T = mkttimes(i);
    
    phi_theta_p = compute_characteristic_function2(1i, T, r0, theta0 + h, k, s, v, m);
    phi_theta_m = compute_characteristic_function2(1i, T, r0, theta0 - h, k, s, v, m);
    P_theta_p = phi_theta_p;
    P_theta_m = phi_theta_m;
    
    ThetaSensitivity_stoch(i) = -(P_theta_p - P_theta_m) / (2 * h * P2);
end

%% Print updated table with sensitivity to theta0
fprintf('\nComparative Table: Duration, Convexity, and Sensitivity to \\theta_0\n');
fprintf('%10s | %20s %20s | %20s %20s | %20s\n', 'Maturity', ...
        'Duration (Stochastic)', 'Duration (Vasicek)', ...
        'Convexity (Stochastic)', 'Convexity (Vasicek)', ...
        'Sensitivity to \\theta_0');

for i = 1:length(mkttimes)
    fprintf('%10.2f | %20.6f %20.6f | %20.6f %20.6f | %20.6f\n', mkttimes(i), ...
        Duration_stoch(i), Duration_vas(i), ...
        Convexity_stoch(i), Convexity_vas(i), ...
        ThetaSensitivity_stoch(i));
end

%% Função característica do Vasicek clássico
function phi_RT = characteristic_function_vasicek(u, T, r0, kappa, sigma, theta_inf)
    s = 1i * u;
    beta_T = s * (1 - exp(-kappa * T)) / kappa;
    alpha_T = -(theta_inf + (s * sigma^2) / (2 * kappa^2)) .* ( beta_T - s * T) ...
              - (sigma^2 / (4 * kappa)) * beta_T.^2;
    phi_RT = exp(alpha_T + beta_T * r0);
end


%%SLTM ChF
function phi_RT = compute_characteristic_function2(u, t, r0, theta0, k, s, v, m)

    
    q = 1i * u;  % função característica (iu)

    % beta_1
    beta1 = (q - q * exp(-k * t)) / k;

    % beta_2
    beta2 = (q * (k - v + v * exp(-k * t))) / (v * (k - v)) ...
            - (k * q * exp(-t * v)) / (v * (k - v));

    % alpha (fórmula simbólica reorganizada)
    term1 = (k * m * q * t) / (k - v) - (m * q * v * t) / (k - v);
    term2 = - (m * q * v * (exp(-k * t) - 1)) / (k * (k - v));
    term3 = (k * m * q * (exp(-v * t) - 1)) / (v * (k - v));

    % termo complexo s^2 * q^2 * (...)
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

    % função característica
    phi = alpha + beta1 * r0 + beta2 * theta0;

    % Proteção contra overflow
    if abs(real(phi)) > 700
        phi_RT = NaN; return;
    end

    phi_RT = exp(phi);

    if ~isfinite(phi_RT) || real(phi_RT) <= 0
        phi_RT = NaN;
    end
end
