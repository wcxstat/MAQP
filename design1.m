N=3; % population
%tau=0.01;
tauvec=[0.01,0.05,0.1,0.3,0.5,0.7,0.9,0.95,0.99];
efpemat=zeros(length(tauvec),8);
for ktau=1:length(tauvec)
tau=tauvec(ktau);
qq=norminv(tau);

nvec=[300,400,300];
%nvec=[300,200,400];
ntest=500;
beta=cell(N,1);
beta{1}=[0.5,0.6,-0.61,-0.48];
beta{2}=[0.5,0.6,-0.61,-0.48];
beta{3}=[0.5,0.6,-0.61,-0.48];
%beta0=[0.5,0.6,-0.61,-0.48];
gamma=cell(N,1);
%gamma{1}=[0.4,0.6,0.5,-0.3,-0.25,1];
gamma{1}=[0.1,0.03,0.12,-0.08,-0.06,0.2];
gamma{2}=[0.49,0.08,0.09,-0.04,-0.06];
gamma{3}=[0.51,0.07,0.1,-0.05,-0.04];
K1=5; % K-fold CV
K2=10;

p=length(beta{1});
predmat=zeros(1000,5);
for rep=1:1000
Ylist=cell(N,1);
XZlist=cell(N,1);

% model 1
p1=p+length(gamma{1});
mu1=zeros(1,p1);
Sigma1=4*toeplitz(0.5.^(0:(p1-1)));
% var=diag([4*ones(1,p),1*ones(1,p1-p)]);
% Sigma1=var.*toeplitz(0.5.^(0:(p1-1)));
XZlist{1}=mvnrnd(mu1,Sigma1,nvec(1));
e=normrnd(0,1,[nvec(1),1]);
% aa1=XZlist{1};
% Ylist{1}=XZlist{1} * [beta{1},gamma{1}]'+sum(abs(aa1(:,1:1)),2).*e;
Ylist{1}=XZlist{1} * [beta{1},gamma{1}]'+e;

% model 2
p2=p+length(gamma{2});
mu=zeros(1,p2);
Sigma=4*toeplitz(0.5.^(0:(p2-1)));
XZlist{2}=mvnrnd(mu,Sigma,nvec(2));
e=normrnd(0,1,[nvec(2),1]);
aa2=XZlist{2};
Ylist{2}=XZlist{2} * [beta{2},gamma{2}]'+sum(aa2(:,1:4).^2,2).*e;
%Ylist{2}=XZlist{2} * [beta{2},gamma{2}]'+e;

% model 3
p3=p+length(gamma{3});
mu=zeros(1,p3);
Sigma=4*toeplitz(0.5.^(0:(p3-1)));
XZlist{3}=mvnrnd(mu,Sigma,nvec(3));
e=normrnd(0,1,[nvec(3),1]);
% aa3=XZlist{3};
% Ylist{3}=XZlist{3} * [beta{3},gamma{3}]'+sum(aa3(:,1:1).^2,2).*e;
Ylist{3}=XZlist{3} * [beta{3},gamma{3}]'+e;

XZnew=mvnrnd(mu1,Sigma1,ntest);
% sigma1=sum(abs(XZnew(:,1:1)),2);
sigma1=ones(ntest,1);
e=normrnd(0,1,[ntest,1]);
Lnew=XZnew * [beta{1},gamma{1}]';
Ynew=Lnew+sigma1.*e;
Qpred_new=Lnew+sigma1*qq;

aic=zeros(1,N);
bic=zeros(1,N);
betahat=zeros(p,N);
XZcov=cell(N,1);
for j=1:N
    XZ=XZlist{j};
    if j==1
        XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):(end-3))];
    else
        XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):end)];
    end
end

for j=1:N
    [coef1,val]=quantreg(XZcov{j},Ylist{j},tau,0);
    pj=size(XZcov{j},2);
    aic(j)=2*nvec(j)*log(val/nvec(j))+2*pj;
    bic(j)=2*nvec(j)*log(val/nvec(j))+log(nvec(j))*pj;
    betahat(:,j)=coef1(1:p);
    if j==1
        gamma_main=coef1((p+1):end);
    end
end

XZnewcov=[XZnew(:,1:p),ones(ntest,1),XZnew(:,(p+1):(end-3))];
Qpred=zeros(ntest,N);
for j=1:N
    Qpred(:,j)=XZnewcov * [betahat(:,j);gamma_main];
end


[coeflist,~]=quantreg_all(XZcov,Ylist,p,tau);
QP_all=XZnewcov*[coeflist{1};coeflist{2}];

w1=CVK(Ylist{1},XZcov{1},1,tau,K1,betahat);
w2=CVK(Ylist{1},XZcov{1},1,tau,K2,betahat);

delta_aic=aic-min(aic);
delta_bic=bic-min(bic);
w_aic=exp(-delta_aic/2)/sum(exp(-delta_aic/2));
w_bic=exp(-delta_bic/2)/sum(exp(-delta_bic/2));
w_sim=ones(N,1)/N;

MAQP1=Qpred * w1;
MAQP2=Qpred * w2;
QP_sim=Qpred * w_sim;
QP_main=Qpred(:,1);
QP_saic=Qpred * w_aic';
QP_sbic=Qpred * w_bic';
mu=-qq*sigma1;
sigma=sigma1;
predmat(rep,1)=mean(EFPE(MAQP1-Qpred_new,mu,sigma,tau));
predmat(rep,2)=mean(EFPE(MAQP2-Qpred_new,mu,sigma,tau));
predmat(rep,3)=mean(EFPE(QP_sim-Qpred_new,mu,sigma,tau));
predmat(rep,4)=mean(EFPE(QP_main-Qpred_new,mu,sigma,tau));
predmat(rep,5)=mean(EFPE(QP_all-Qpred_new,mu,sigma,tau));
predmat(rep,6)=mean(EFPE(QP_saic-Qpred_new,mu,sigma,tau));
predmat(rep,7)=mean(EFPE(QP_sbic-Qpred_new,mu,sigma,tau));
end
aa=mean(predmat);
gain=(min(aa(3:end))-min(aa(1:2)))/min(aa(2:end));
efpemat(ktau,:)=[aa,gain];
end

save('result1.2.txt','efpemat','-ascii')