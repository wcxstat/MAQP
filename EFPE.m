function value=EFPE(diff,mu,sigma,tau)
n0=length(diff);
value=zeros(1,n0);
for i=1:n0
    if diff(i)>0
        arg=linspace(0,diff(i),200);
        Fdiff=cdf('Normal',arg,mu(i),sigma(i))-tau;
    else
        arg=linspace(diff(i),0,200);
        Fdiff=-cdf('Normal',arg,mu(i),sigma(i))+tau;
    end
    value(i)=trapz(arg,Fdiff);
end