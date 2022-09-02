function [llobj,llobji] = loglikelihood2(param,inp)

theta0 = param(1,1);
theta1 = param(2,1);
theta2 = param(3,1);
theta3 = param(4,1);
theta4 = param(5,1);
theta5 = param(6,1);

rho = param(7,1);
sigeta = param(8,1);

x1 = inp.x1;
x2 = inp.x2;
z = inp.z;
n = inp.n;
y = inp.y;

%% p1
X1 = [ones(n,1),x1,z]; %First stage explanatory vars

fitx2 = ones(n,1)*theta3 + x1*theta4 + z*theta5;
eta = x2-fitx2;

% The distribution that eta follows is N(0,sigeta^2)
px2 = max(normpdf(eta,0,sigeta), 1.0e-100);
logpx2 = log(px2);
firstterm = sum(logpx2);

%% p2
X2 = [ones(n,1),x1,x2];

fityi = ones(n,1)*theta0 + x1*theta1 + x2*theta2;

% The distribution that epsilon follows is N((rho/(sigeta^2))*eta,1-(rho^2/sigeta^2))

meaneps = (rho/(sigeta^2))*eta;
vareps = abs(1-(rho^2/sigeta^2));
stdeveps = sqrt(vareps);
logpx1= y.*log(max(cdf('Normal',fityi,meaneps,stdeveps),1.0e-100))+(ones(n,1)-y).*log(max(1-cdf('Normal',fityi,meaneps,stdeveps),1.0e-100));
secondterm = sum(logpx1);

logp = firstterm+secondterm;

llobji = logpx1 + logpx2;
llobj = - logp/n;

end











