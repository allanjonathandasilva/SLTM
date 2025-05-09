clc
clear
format long
r0 = 0.1;
kappa = 0.1265;
theta = 0.0802;
sigma = 0.0218;
y0 = 100000;

l = 1;
L = 5;
e = 0.005;
strike = 145000;

q = -1i;
r0 =  0.149065331170911;
theta0 = 0.022004688009579;%0.1;
m = 0.121353131400855;%0.15;
s = 0.010000000001806;%0.015;
k = 1.239226307246909;%0.25;
v = 1.237519990864449-2*0.2;%0.5;
t = 5;
z=t;
for j = 1:3
v = v+0.2
strike = 145000;
for i=1:46


A = @(x) (m*q*v*x*z)/(k - v) - (k*m*q*x*z)/(k - v) - (s.^2*((q.^2*x.^2*(3*k.^4 - 4*k.^4*exp(-v*z) + k.^4*exp(-2*v*z)))/2 + (q.^2*v.^3*x.^2*(k*exp(-2*k*z) - k + 2*k.^2*z))/2 - (q.^2*v*x.^2*(2*k.^4*z + k.^3 - k.^3*exp(-2*v*z)))/2 + (q.^2*v.^2*x.^2*(2*k.^3*z - 4*k.^2*exp(- k*z - v*z) - 4*k.^2 + 4*k.^2*exp(-k*z) + 4*k.^2*exp(-v*z)))/2 - (q.^2*v.^4*x.^2*(4*exp(-k*z) - exp(-2*k*z) + 2*k*z - 3))/2))/(2*k*v.^3*(k + v)*(k - v).^2) - (k*m*q*x*(exp(-v*z) - 1))/(v*(k - v)) + (m*q*v*x*(exp(-k*z) - 1))/(k*(k - v));
B1 = @(x) -(q.*x - q.*x.*exp(-k.*t))./k;
B2 = @(x)  (k*q*x*exp(-t*v))/(v*(k - v)) - (q*x*(k - v + v*exp(-k*t)))/(v*(k - v));   %(q.*x.*(k - v + v.*exp(-k.*t)))./(v.*(k - v)) - (k.*q.*x.*exp(-t.*v))./(v.*(k - v));
           

cf = @(x)  exp(A(x)+B1(x).*r0+B2(x).*theta0);

% cf = @(s) exp( -(theta-(-1i*s.*sigma^2/(2*kappa^2))).*(((-1i*s./kappa).*(exp(-kappa*t)-1))-1i*s.*t) -sigma^2/(4*kappa)*((-1i*s./kappa).*(exp(-kappa*t)-1)).^2+(-L*t+(L./(kappa+1i*s*e)).*log((1+e*((-1i*s/kappa)*(exp(-kappa*t)-1)))/(exp(-kappa*t))))  + ((-1i*s./kappa).*(exp(-kappa*t)-1)).*r0 );
f(i,1,j) = COS_method1(cf,5000,-0.397350611302976,1.497317858897704,strike);
% C(i,1,j) = fajardo_new(r0, y0, t, kappa, theta, sigma, strike);
sig(i,1,j) = BlackVol('C', y0, strike, t, r0, f(i,1,j));
strike = strike + 1000;
end
end
str = 145000:1000:190000;
figure, hold on
% plot(str,sig,'r-o')
 plot(str,sig(:,1,1),'b-o',str,sig(:,1,2),'r-o',str,sig(:,1,3),'k-o')
xlabel('Strikes')
title('Black Implied Volatilities varying $v$','Interpreter','latex')
hold off