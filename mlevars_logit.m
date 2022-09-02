function vars_matrix = mlevars_logit(param0,inp)

stpsize = 0.00001;

X = inp.X1;
n = inp.n;
y = inp.y;

deltalni_3 = zeros(n,3);

for coeff = 1:3
    
    param = param0;
    param(coeff,1) = param0(coeff,1)*(1+stpsize); % Change param values
    
    Xfit0 = X*param0;
    
    epspb0 = (ones(n,1)-y).*max(1-cdf('Logistic',Xfit0,0,1),0.00001)+y.*max(cdf('Logistic',Xfit0,0,1),0.00001);
    llobji0 = log(epspb0);
    
    Xfit = X*param;
    epspb = (ones(n,1)-y).*max(1-cdf('Logistic',Xfit,0,1),0.00001)+y.* max(cdf('Logistic',Xfit,0,1),0.00001);
    llobji = log(epspb);
    
    
    deltalni = (llobji - llobji0)/(stpsize*param0(coeff,1));
    
    deltalni_3(:,coeff) = deltalni;
end

stack_indiv_var = zeros(3,3,n);

for i = 1:n
    stack_indiv_var(:,:,i) = deltalni_3(i,:)'*deltalni_3(i,:);
end

sum_indiv_var = sum(stack_indiv_var,3);

vars_matrix = inv(sum_indiv_var);