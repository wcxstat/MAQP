N=3; % population
%tau=0.05;
tauvec=[0.01,0.05,0.1,0.3,0.5,0.7,0.9,0.95,0.99];
efpemat=zeros(length(tauvec),8);
wmean1=zeros(length(tauvec),4);
wstd1=zeros(length(tauvec),4);
wmean2=zeros(length(tauvec),4);
wstd2=zeros(length(tauvec),4);
for ktau=1:length(tauvec)
tau=tauvec(ktau);
qq=norminv(tau);

nvec=[100,200,100];
ntest=500;
beta0=[0.5,0.6,-0.61,-0.48];
gamma=cell(N,1);
%gamma{1}=[0.4,0.6,0.5,-0.3,-0.25];
gamma{1}=[0.1,0.03,0.12,-0.08,-0.06,0.2];
gamma{2}=[0.49,0.08,0.09,-0.04,-0.06];
gamma{3}=[0.51,0.07,0.1,-0.05,-0.04];
K1=5; % K-fold CV
K2=10;
p=length(beta0);

predmat=zeros(1000,5);
weight1=zeros(N,1000);
weight2=zeros(N,1000);
for rep=1:1000
Ylist=cell(N,1);
XZlist=cell(N,1);

% model 1
p1=p+length(gamma{1});
mu1=zeros(1,p1);
Sigma1=4*toeplitz(0.5.^(0:(p1-1)));
%var=diag([4*ones(1,p),0.1*ones(1,p1-p)]);
%Sigma1=var.*toeplitz(0.5.^(0:(p1-1)));
XZlist{1}=mvnrnd(mu1,Sigma1,nvec(1));
e=normrnd(0,1,[nvec(1),1]);
Ylist{1}=XZlist{1} * [beta0,gamma{1}]'+e;

% model 2
p2=p+length(gamma{2});
mu=zeros(1,p2);
Sigma=4*toeplitz(0.5.^(0:(p2-1)));
XZlist{2}=mvnrnd(mu,Sigma,nvec(2));
e=normrnd(0,1,[nvec(2),1]);
aa=XZlist{2};
Ylist{2}=XZlist{2} * [beta0,gamma{2}]'+sum(aa(:,1:4).^2,2).*e;

% model 3
p3=p+length(gamma{3});
mu=zeros(1,p3);
Sigma=4*toeplitz(0.5.^(0:(p3-1)));
XZlist{3}=mvnrnd(mu,Sigma,nvec(3));
e=normrnd(0,1,[nvec(3),1]);
Ylist{3}=XZlist{3} * [beta0,gamma{3}]'+e;

XZnew=mvnrnd(mu1,Sigma1,ntest);
e=normrnd(0,1,[ntest,1]);
Lnew=XZnew * [beta0,gamma{1}]';
Ynew=Lnew+e;
Qpred_new=Lnew+qq;

aic=zeros(1,N);
bic=zeros(1,N);
betahat=zeros(p,N);
XZcov=cell(N,1);
% for j=1:N
%     XZ=XZlist{j};
%     if (j==1)||(j==3)
%         XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):end)];
%     else
%     XZcov{j}=XZ;
%     end
% end
for j=1:N
    XZ=XZlist{j};
    XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):end)];
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

XZnewcov=[XZnew(:,1:p),ones(ntest,1),XZnew(:,(p+1):end)];
Qpred=zeros(ntest,N);
for j=1:N
    Qpred(:,j)=XZnewcov * [betahat(:,j);gamma_main];
end

[coeflist,~]=quantreg_all(XZcov,Ylist,p,tau);
QP_all=XZnewcov*[coeflist{1};coeflist{2}];

w1=CVK(Ylist{1},XZcov{1},1,tau,K1,betahat);
w2=CVK(Ylist{1},XZcov{1},1,tau,K2,betahat);
weight1(:,rep)=w1;
weight2(:,rep)=w2;

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
mu=-qq*1*ones(1,ntest);
sigma=1*ones(1,ntest);
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

wmean1(ktau,:)=[mean(weight1,2)',mean(sum(weight1([1,3],:),1))];
wstd1(ktau,:)=[std(weight1'),std(sum(weight1([1,3],:),1))];
wmean2(ktau,:)=[mean(weight2,2)',mean(sum(weight2([1,3],:),1))];
wstd2(ktau,:)=[std(weight2'),std(sum(weight2([1,3],:),1))];
end

save('result3.1.txt','efpemat','-ascii')
save('result3.1wmean.txt','wmean1','-ascii')
save('result3.1wstd.txt','wstd1','-ascii')
