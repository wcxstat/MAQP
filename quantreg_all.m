function [coeflist,vel]=quantreg_all(xlist,ylist,p,tau)

N=length(ylist);
nvec=zeros(1,N);
qvec=zeros(1,N);
for j=1:N
    nvec(j)=length(ylist{j});
    qvec(j)=size(xlist{j},2)-p;
end
nsum=sum(nvec);
f=[zeros(sum(qvec)+p,1);tau*ones(nsum,1);(1-tau)*ones(nsum,1)];

A=[zeros(2*nsum,sum(qvec)+p),-eye(2*nsum)];
b=zeros(2*nsum,1);
beq=zeros(nsum,1);
Aeq1=zeros(nsum,sum(qvec)+p);
for j=1:N
    xj=xlist{j};
    index1=(1+sum(nvec(1:(j-1)))):sum(nvec(1:j));
    index2=(p+1+sum(qvec(1:(j-1)))):(sum(qvec(1:j))+p);
    Aeq1(index1,1:p)=xj(:,1:p);
    Aeq1(index1,index2)=xj(:,(p+1):end);
    beq(index1)=ylist{j};
end
Aeq=[Aeq1,eye(nsum),-eye(nsum)];
[coef,vel]=linprog(f,A,b,Aeq,beq);
coeflist=cell(N+1,1);
coeflist{1}=coef(1:p);
for j=1:N
    index2=(p+1+sum(qvec(1:(j-1)))):(sum(qvec(1:j))+p);
    coeflist{j+1}=coef(index2);
end
end