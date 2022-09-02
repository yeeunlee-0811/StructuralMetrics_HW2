function llobj = loglikelihood(param,inp)

X = inp.X1;
n = inp.n;
y = inp.y;

v1 = X*param; % n-by-1

llobji = zeros(n,1);

for i =1:n
    yi = y(i,1);
    if yi == 0
        epsct = v1(i,1);
        llobji(i,1) = max(1-cdf('Normal',epsct,0,1),0.00001);
    else
        epsct = v1(i,1);
        llobji(i,1) = max(cdf('Normal',epsct,0,1),0.00001);
    end
end


llobji0 = log(llobji);

llobj = -sum(llobji0);
    


        
