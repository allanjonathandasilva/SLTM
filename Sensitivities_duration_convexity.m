clear all;
clc;

%% Table 3: Yield, Duration, and Convexity under a Stochastic Mean-Reverting Model

% Model parameters
r0      = 0.149;     % Initial short rate
theta0  = 0.022;     % Initial long-term mean
s       = 0.01;      % Volatility of theta(t)
v       = 1.2375;    % Mean-reversion speed of theta(t)
k       = 1.2392;    % Mean-reversion speed of r(t)
m       = 0.1213;    % Long-run level of theta(t)
h       = 1e-4;      % Finite difference increment

% Maturities (in years) for analysis
mkttimes = [1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9]';

% Header for tabular output
fprintf('\nTable 3: Yield, Duration, and Convexity (Stochastic Long-Term Mean Model)\n');
fprintf('%10s | %12s %12s %12s\n', 'Maturity', 'Yield (%)', 'Duration', 'Convexity');

% Compute bond metrics for each maturity
for i = 1:length(mkttimes)
    T = mkttimes(i);
    
    % Evaluate characteristic function at base and perturbed values of r0
    phi    = compute_characteristic_function2(1i, T, r0,      theta0, k, s, v, m);
    phi_p  = compute_characteristic_function2(1i, T, r0 + h,  theta0, k, s, v, m);
    phi_m  = compute_characteristic_function2(1i, T, r0 - h,  theta0, k, s, v, m);
    
    % Extract price and numerical derivatives
    P  = real(phi); 
    Pp = real(phi_p); 
    Pm = real(phi_m);
    
    % Compute yield, duration, and convexity
    Y = -log(P) / T;                          % Continuously compounded yield
    D = -(Pp - Pm) / (2 * h * P);             % Numerical first derivative (duration)
    C = (Pp - 2 * P + Pm) / (h^2 * P);        % Numerical second derivative (convexity)
    
    % Print formatted row
    fprintf('%10.2f | %12.4f %12.4f %12.4f\n', T, 100 * Y, D, C);
end

%% Closed-form characteristic function: ∫₀^t r(s) ds under stochastic mean model
function phi_RT = compute_characteristic_function2(u, t, r0, theta0, k, s, v, m)
    % Computes the closed-form characteristic function of the integrated short rate
    % x(t) = ∫₀^t r(s) ds, under the model:
    %   dr(t)     = k*(theta(t) - r(t)) dt
    %   dtheta(t) = v*(m - theta(t)) dt + s*dW(t)
    % Input:
    %   u       - Complex argument for characteristic function (usually 1i*frequency)
    %   t       - Time to maturity
    %   r0      - Initial short rate
    %   theta0  - Initial long-term mean
    %   k       - Mean-reversion speed of r(t)
    %   s       - Volatility of theta(t)
    %   v       - Mean-reversion speed of theta(t)
    %   m       - Long-term mean level of theta(t)
    % Output:
    %   phi_RT  - Characteristic function value at u

    q = 1i * u;  % Fourier variable

    % Affine coefficient beta_1
    beta1 = (q - q * exp(-k * t)) / k;

    % Affine coefficient beta_2
    beta2 = (q * (k - v + v * exp(-k * t))) / (v * (k - v)) ...
            - (k * q * exp(-v * t)) / (v * (k - v));

    % Drift component (alpha) as derived from analytical expansion
    term1 = (k * m * q * t) / (k - v) - (m * q * v * t) / (k - v);
    term2 = - (m * q * v * (exp(-k * t) - 1)) / (k * (k - v));
    term3 = (k * m * q * (exp(-v * t) - 1)) / (v * (k - v));

    % Second-order correction due to stochastic volatility (s^2 q^2 term)
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

    % Full characteristic exponent
    phi = alpha + beta1 * r0 + beta2 * theta0;

    % Overflow protection (exponential threshold)
    if abs(real(phi)) > 700
        phi_RT = NaN; return;
    end

    phi_RT = exp(phi);

    if ~isfinite(phi_RT) || real(phi_RT) <= 0
        phi_RT = NaN;
    end
end
