clear

%% Importing data
load ps2.dat
load sim3.dat

data = ps2;
y = data(:,1); 

x1 = data(:,2);
x2 = data(:,3);
z = data(:,4);
n = size(data,1);

inp.data = data;
inp.y =y;
inp.x1 = x1;
inp.x2 = x2;
inp.z = z;
inp.n = n;
%% Q1 - (1): Using ML, get the parameters and stdev
% Non-invertible but simple version

X1 = [ones(n,1), x1, x2];
k1 = size(X1,2);
inp.X1 = X1;
inp.k1 = k1;
%% Loglikelihood_Normal
% Parameter

llobj = @(param)loglikelihood(param,inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval);

param0 = zeros(k1,1);
[mleparam,logval] = fminsearch(llobj,param0,options);

% Standard Errors
vars_matrix = mlevars(mleparam,inp);
mle_se = sqrt(diag(vars_matrix));

%% average woman 
avgage = mean(x1); % avgage = 42.5378
avged = mean(x2);  % avged = 12.2869

avgx = [1,avgage, avged];
avgxed = [1,avgage, avged+1]; % an additional year of education

pavg = cdf('Normal',avgx*mleparam,0,1); % pavg = 0.5705
pavg_ed1 = cdf('Normal',avgxed*mleparam,0,1); % pavg_ed1 = 0.6109

%% Loglikelihood_Logistic

llobj_logit = @(param)loglikelihood_logit(param,inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval);

param0 = zeros(k1,1);
[mleparam_logit,logval_logit] = fminsearch(llobj_logit,param0,options);

% Standard Errors
vars_matrix_logit = mlevars_logit(mleparam_logit,inp);
mle_se_logit = sqrt(diag(vars_matrix_logit));

%% Q2_d

% ML -> Need likelihood p(ytilde|explanatory vars)
k2 = 8;
inp.k2 =k2;

rho =0.7;
sigeta = 1;

mleparam2mat = zeros(k2,10);
logval2mat = zeros(10,1);

q2param0 = [-1.79364175497898;-0.0101846047078158;0.197007764175514;0.197276817688595;0.0510832173032688;1.20476098225882;0.0967021989739409;3.11078283870299];

llobj2 = @(param) loglikelihood2(param,inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10);

[mleparam2,logval2] = fminsearch(llobj2,q2param0,options);

%save('mleparam2.mat','mleparam2');

%% 
load mleparam2

vars_matrix2 = mlevars2(mleparam2,inp);
mle_se2 = sqrt(diag(vars_matrix2));


%% Q3-d

data3 = sim3;
cid = data3(:,1);
week = data3(:,2);
yit = data3(:,3);
pit = data3(:,4);
n3 = size(data3,1);

uniqweek = unique(week);
uniqi = unique(cid);

nweek = size(uniqweek,1);
ni = size(uniqi,1);

inp.data3= data3;
inp.cid =cid;
inp.week =week;
inp.yit = yit;
inp.pit = pit;
inp.n3=n3;
inp.nweek = nweek;
inp.ni = ni;
inp.uniqweek = uniqweek;
inp.uniqi = uniqi;

% Draws. Fix after generation
S = 1000;
inp.S = S;
%epsdraw = randn(ni,S);
%save('epsdraw.mat','epsdraw')
load epsdraw.mat

inp.epsdraw = epsdraw;

%% q3_d
k3 = 4;
inp.k3 =k3;
llobj3 = @(param) loglikelihood3(param,inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10);

q3param0 = randn(k3,1);
[mleparam3,logval3] = fminsearch(llobj3,q3param0,options);

vars_matrix_q3 = mlevars3(mleparam3,inp);
se_q3 = sqrt(diag(vars_matrix_q3));

%% q3-f
k3 = 3;
inp.k3 =k3;

llobj3f = @(param) loglikelihoodf(param,inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10);

q3param0 = randn(k3,1);
[mleparam3f,logval3f] = fminsearch(llobj3f,q3param0,options);
%%
vars_matrix_q3f = mlevars3f(mleparam3f,inp);
se_q3f = sqrt(diag(vars_matrix_q3f));
