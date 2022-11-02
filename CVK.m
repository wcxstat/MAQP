function [w]=CVK(mainY,mainXZ,mainj,tau,K,betahat)

% Input:
% mainY: response from the main model, n1-by-1 vector
% mainXZ: covariate from the main model, n1-by-p matrix
% mainj: which model is the main model
% tau: quantile level
% K: K-fold cross validation
% betahat: p1-by-N estimated coefficient matrix

% Output:
% w: selected weights (N by 1 vector)

n1=length(mainY); % sample size for the main sample
[p1,N]=size(betahat);
M=floor(n1/K); % sample size for each group
Qhat=zeros(n1,N);
for k=1:K
    if k<K
        index=(k-1)*M+(1:M);
    else
        index=(K*M-M+1):n1;
    end
    index1=setdiff(1:n1,index);
    mainY_k=mainY(index1);
    mainXZ_k=mainXZ(index1,:);
    [coef1,~]=quantreg(mainXZ_k,mainY_k,tau,0);
    for j=1:N
        if j==mainj
            Qhat(index,j)=mainXZ(index,:)*coef1;
        else
            Qhat(index,j)=mainXZ(index,:)*[betahat(:,j);coef1((p1+1):end)];
        end
    end
end

f=[zeros(N,1);tau*ones(n1,1);(1-tau)*ones(n1,1)];
Aeq=[[Qhat,eye(n1),-eye(n1)];[ones(1,N),zeros(1,n1),zeros(1,n1)]];
slt=linprog(f,[],[],Aeq,[mainY;1],zeros(N+2*n1,1),[]);
w=slt(1:N);