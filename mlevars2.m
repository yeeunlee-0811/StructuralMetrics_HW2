function vars_matrix = mlevars2(param0,inp)

stpsize = 0.1;


n = inp.n;
k2 = inp.k2;

deltalni_3 = zeros(n,k2);

for coeff = 1:k2
    
    param = param0;
    param(coeff,1) = param0(coeff,1)*(1+stpsize); % Change param values
    
    [~,llobji0] = loglikelihood2(param0,inp);
    
    [~,llobji] = loglikelihood2(param,inp);
    
    
    deltalni = (llobji - llobji0)/(stpsize*param0(coeff,1));
    
    deltalni_3(:,coeff) = deltalni;
end

stack_indiv_var = zeros(k2,k2,n);

for i = 1:n
    stack_indiv_var(:,:,i) = deltalni_3(i,:)'*deltalni_3(i,:);
end

sum_indiv_var = sum(stack_indiv_var,3);

vars_matrix = inv(sum_indiv_var);
end