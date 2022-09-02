function [llobj,logli]= loglikelihood3(param,inp)

epsdraw = inp.epsdraw;
yit =inp.yit;
cid = inp.cid;
week = inp.week;
pit = inp.pit;
S = inp.S;
ni = inp.ni;
nweek = inp.nweek;
uniqweek = inp.uniqweek;
uniqi = inp.uniqi;

% reshape yit matrix: indiv x nweek

matyit = reshape(yit,nweek,ni)';
matyit = [zeros(ni,1),matyit];

% reshpae pit matrix: indiv x nweek
matpit = reshape(pit,nweek,ni)';
matpit = [zeros(ni,1),matpit];



pits = zeros(ni,nweek,S);

for t = 2:nweek+1
    pt = matpit(:,t);
    lagyt = matyit(:,t-1);
    yt = matyit(:,t);
    for s = 1:S
        eps = epsdraw(:,s);
        
        v1 = [ones(ni,1),pt,lagyt,eps]*param;
        v2 = exp(v1);
        v3 = v2./(ones(ni,1)+v2);
        
        v4 = yt.*v3 +(ones(ni,1)-yt).*(ones(ni,1)./(ones(ni,1)+v2));
        
        pits(:,t-1,s) = yt.*v3 +(ones(ni,1)-yt).*(ones(ni,1)./(ones(ni,1)+v2));
    end
end

pis = prod(pits,2); % pis should be ni x 1 x S

li = mean(pis,3);

logli = log(max(li,0.00000001));

llobj = -sum(logli);

end
