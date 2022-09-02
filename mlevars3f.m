function vars_matrix = mlevars3f(param0,inp)

stpsize = 0.00001;

k3 = inp.k3;
ni = inp.ni;

deltalni_k = zeros(ni,k3);

for k =1:k3
    param = param0;
    param(k,1) = param0(k,1)*(1+stpsize); % Change param values
    
    [~,pyihat] = loglikelihoodf(param,inp);
    [~,pyihat0] = loglikelihoodf(param0,inp);
    
    deltalni = (pyihat - pyihat0)/(stpsize*param0(k,1));
    
    deltalni_k(:,k) = deltalni;
end

stack_indiv_var = zeros(k3,k3,ni);

for i = 1:ni
    stack_indiv_var(:,:,i) = deltalni_k(i,:)'*deltalni_k(i,:);
end

sum_indiv_var = sum(stack_indiv_var,3);

vars_matrix = inv(sum_indiv_var);