N=7; % population
%tau=0.05;
tauvec=[0.01,0.05,0.1,0.3,0.5,0.7,0.9,0.95,0.99];
efpemat=zeros(length(tauvec),8);
wmean1=zeros(length(tauvec),N);
wstd1=zeros(length(tauvec),N);
wmean2=zeros(length(tauvec),N);
wstd2=zeros(length(tauvec),N);
weight1mat=zeros(N*length(tauvec),1000);
weight2mat=zeros(N*length(tauvec),1000);
for ktau=1:length(tauvec)
tau=tauvec(ktau);
qq=norminv(tau);

nvec=[100,200,100,200,100,200,100];
%nvec=[300,400,300,400,300,400,300];
%nvec=500*ones(1,N);
ntest=500;
beta=cell(N,1);
beta{1}=[0.5,0.6,-0.61,-0.48];
beta{2}=[0.5,0.6,-0.61,-0.48];
beta{3}=[0.5,0.6,-0.61,-0.48]+0.01;
beta{4}=[0.5,0.6,-0.61,-0.48];
beta{5}=[0.5,0.6,-0.61,-0.48];
beta{6}=[0.5,0.6,-0.61,-0.48]+1;
beta{7}=[0.5,0.6,-0.61,-0.48]-0.01;
gamma=cell(N,1);
%gamma{1}=[0.4,0.6,0.5,-0.3,-0.25];
gamma{1}=[0.1,0.03,0.12,-0.08,-0.06,0.2];
gamma{2}=[0.49,0.08,0.09,-0.04,-0.06,2.5];
gamma{3}=[0.51,0.07,0.1,-0.05,-0.04];
gamma{4}=[0.02,0.05,-0.03,-0.01,2.5];
gamma{5}=[0.03,-0.07,0.06,0.02];
gamma{6}=[-0.85,2.5];
gamma{7}=-0.87;
K1=5; % K-fold CV
K2=10;
p=length(beta{1});

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
Ylist{1}=XZlist{1} * [beta{1},gamma{1}]'+e;

% model 2
p2=p+length(gamma{2});
mu=zeros(1,p2);
Sigma=4*toeplitz(0.5.^(0:(p2-1)));
XZlist{2}=mvnrnd(mu,Sigma,nvec(2));
e=normrnd(0,1,[nvec(2),1]);
aa=XZlist{2};
Ylist{2}=XZlist{2} * [beta{2},gamma{2}]'+sum(aa(:,1:4).^2,2).*e;

% model 3
p3=p+length(gamma{3});
mu=zeros(1,p3);
Sigma=4*toeplitz(0.5.^(0:(p3-1)));
XZlist{3}=mvnrnd(mu,Sigma,nvec(3));
e=normrnd(0,1,[nvec(3),1]);
Ylist{3}=XZlist{3} * [beta{3},gamma{3}]'+e;

% model 4
p4=p+length(gamma{4});
mu=zeros(1,p4);
Sigma=4*toeplitz(0.5.^(0:(p4-1)));
XZlist{4}=mvnrnd(mu,Sigma,nvec(4));
e=normrnd(0,1,[nvec(4),1]);
aa=XZlist{4};
Ylist{4}=XZlist{4} * [beta{4},gamma{4}]'+sum(aa(:,1:4).^2,2).*e;

% model 5
p5=p+length(gamma{5});
mu=zeros(1,p5);
Sigma=4*toeplitz(0.5.^(0:(p5-1)));
XZlist{5}=mvnrnd(mu,Sigma,nvec(5));
e=normrnd(0,1,[nvec(5),1]);
Ylist{5}=XZlist{5} * [beta{5},gamma{5}]'+e;

% model 6
p6=p+length(gamma{6});
mu=zeros(1,p6);
Sigma=4*toeplitz(0.5.^(0:(p6-1)));
XZlist{6}=mvnrnd(mu,Sigma,nvec(6));
e=normrnd(0,1,[nvec(6),1]);
Ylist{6}=XZlist{6} * [beta{6},gamma{6}]'+e;

% model 7
p7=p+length(gamma{7});
mu=zeros(1,p7);
Sigma=4*toeplitz(0.5.^(0:(p7-1)));
XZlist{7}=mvnrnd(mu,Sigma,nvec(7));
e=normrnd(0,1,[nvec(7),1]);
Ylist{7}=XZlist{7} * [beta{7},gamma{7}]'+e;

% new dataset from model 1
XZnew=mvnrnd(mu1,Sigma1,ntest);
sigma1=ones(1,ntest);
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
    XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):end)];
end
% for j=1:N
%     XZ=XZlist{j};
%     if j==1
%         XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):(end-3))];
%     else
%         XZcov{j}=[XZ(:,1:p),ones(nvec(j),1),XZ(:,(p+1):end)];
%     end
% end
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
bb=mean(predmat);
gain=(min(bb(3:end))-min(bb(1:2)))/min(bb(2:end));
efpemat(ktau,:)=[bb,gain];

wmean1(ktau,:)=mean(weight1,2)';
wstd1(ktau,:)=std(weight1');
wmean2(ktau,:)=mean(weight2,2)';
wstd2(ktau,:)=std(weight2');
index=(1+(ktau-1)*N):(ktau*N);
weight1mat(index,:)=weight1;
weight2mat(index,:)=weight2;
end

save('result4.1.txt','efpemat','-ascii')
save('result4.1wm1.txt','wmean1','-ascii')
save('result4.1wm2.txt','wmean2','-ascii')
save('result4.1wstd1.txt','wstd1','-ascii')
save('result4.1wstd2.txt','wstd2','-ascii')
save('result4.1w1.txt','weight1mat','-ascii')
save('result4.1w2.txt','weight2mat','-ascii')