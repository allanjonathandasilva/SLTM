clear all;
clc;

%% Sensitivity Table C4: Impact of theta0 across varying model parameters (s, v, kappa)

% Initial conditions
r0      = 0.149;     % Initial short rate
theta0  = 0.022;     % Initial long-term mean
m       = 0.1213;    % Long-term average of the mean-reverting level
T       = [1, 5, 9]; % Selected maturities of interest (in years)
h       = 1e-4;      % Finite difference increment

% Grid of parameter values for sensitivity analysis
s_values     = [0.005, 0.01, 0.02];     % Volatility values of theta(t)
v_values     = [0.8, 1.2, 2.0];         % Mean-reversion speeds of theta(t)
kappa_values = [0.5, 1.0, 1.5];         % Mean-reversion speeds of r(t)

fprintf('Sensitivity Table: Price Sensitivity to \\theta_0 under varying (s, v, \\kappa)\n');
fprintf('%8s %8s %8s | %15s %15s %15s\n', 's', 'v', 'kappa', ...
        'Sensitivity @ 1y', 'Sensitivity @ 5y', 'Sensitivity @ 9y');

% Loop over the grid of parameters
for s = s_values
    for v = v_values
        for k = kappa_values
            for j = 1:length(T)
                % Characteristic function evaluations for central and perturbed theta0
                phi     = compute_characteristic_function2(1i, T(j), r0, theta0, k, s, v, m);
                phi_p   = compute_characteristic_function2(1i, T(j), r0, theta0 + h, k, s, v, m);
                phi_m   = compute_characteristic_function2(1i, T(j), r0, theta0 - h, k, s, v, m);
                
                % Real part extraction for pricing (if needed)
                P       = real(phi); 
                Pp      = real(phi_p); 
                Pm      = real(phi_m);
                
                % Central difference approximation of partial derivative dP/dtheta0
                S(j)    = -(Pp - Pm) / (2 * h * P);
            end
            % Display results for current parameter configuration
            fprintf('%8.4f %8.4f %8.4f | %15.6f %15.6f %15.6f\n', s, v, k, S(1), S(2), S(3));
        end
    end
end

%% ChF - Sec 3.3, thm 1
function phi_RT = compute_characteristic_function2_old(u, T, r0, theta0, k, s, v, m)
    q = 1i * u;
    t = T; z = T;
    exp_kz = exp(k * z);
    exp_vz = exp(v * z);
    exp_kzvz = exp_kz * exp_vz;

    % Compute the exponential-affine term A(T)
    A_num = q * exp(-k * z) * exp(-v * z) * (s^2 + 2 * m * v) * ...
        (k^2 * exp_kz - v^2 * exp_vz - k^2 * exp_kzvz + v^2 * exp_kzvz ...
         - k * v^2 * z * exp_kzvz + k^2 * v * z * exp_kzvz);
    A_den = 2 * k * v^2 * (k - v);
    A = A_num / A_den;

    % Compute affine coefficients
    B1 = (q - q * exp(-k * t)) / k;
    B2 = (q * (k - v + v * exp(-k * t))) / (v * (k - v)) ...
         - (k * q * exp(-v * t)) / (v * (k - v));

    % Return the characteristic function
    phi_RT = exp(A + B1 * r0 + B2 * theta0);
end

%% ChF - Sec 3.3, thm 1
function phi_RT = compute_characteristic_function2(u, t, r0, theta0, k, s, v, m)
    % Closed-form ChF for accumulated stochastic interest rate
    % Inputs:
    %   u       - Complex argument (typically 1i * real frequency)
    %   t       - Maturity time
    %   r0      - Initial short rate
    %   theta0  - Initial long-term mean
    %   k       - Mean-reversion rate of r(t)
    %   s       - Volatility of theta(t)
    %   v       - Mean-reversion rate of theta(t)
    %   m       - Long-term mean level of theta(t)
    % Output:
    %   phi_RT  - Characteristic function evaluated at u

    q = 1i * u;  % ChF argument (iu)

    % Coefficient beta_1 Eq. 30
    beta1 = (q - q * exp(-k * t)) / k;

    % Coefficient beta_2 Eq. 31
    beta2 = (q * (k - v + v * exp(-k * t))) / (v * (k - v)) ...
            - (k * q * exp(-v * t)) / (v * (k - v));

    % Coefficient alpha 
    term1 = (k * m * q * t) / (k - v) - (m * q * v * t) / (k - v);
    term2 = - (m * q * v * (exp(-k * t) - 1)) / (k * (k - v));
    term3 = (k * m * q * (exp(-v * t) - 1)) / (v * (k - v));

    
    q2 = q^2;
    num_s = ...
        (q2 * (3 * k^4 - 4 * k^4 * exp(-v * t) + k^4 * exp(-2 * v * t))) / 2 ...
      + (q2 * v^3 * (k * exp(-2 * k * t) - k + 2 * k^2 * t)) / 2 ...
      - (q2 * v * (2 * k^4 * t + k^3 - k^3 * exp(-2 * v * t))) / 2 ...
      + (q2 * v^2 * (2 * k^3 * t - 4 * k^2 * exp(-k * t - v * t) ...
         - 4 * k^2 + 4 * k^2 * exp(-k * t) + 4 * k^2 * exp(-v * t))) / 2 ...
      - (q2 * v^4 * (4 * exp(-k * t) - exp(-2 * k * t) + 2 * k * t - 3)) / 2;

    denom_s = 2 * k * v^3 * (k + v) * (k - v)^2;

    % Solution of Eq. 27 given by Eq. 32
    alpha = term1 + term2 + term3 - (s^2 * num_s) / denom_s;

    % ChF expression - Solution of Eqs. 27, 28 and 29
    phi = alpha + beta1 * r0 + beta2 * theta0;

    % Overflow protection
    if abs(real(phi)) > 700
        phi_RT = NaN;
        return;
    end
    
    % Eq. 26
    phi_RT = exp(phi);

    if ~isfinite(phi_RT) || real(phi_RT) <= 0
        phi_RT = NaN;
    end
end
