diary('CCV.txt')
diary on;
close all;
clear all;
clc

warning off;
num_p=3;
addpath('.\datasets');
addpath('.\functions');
dataname = 'CCV';

fprintf('----------- Database:【 %s 】---------------\n',dataname);
load([dataname '.mat']);
% X = fea;
% Y = gt;
X1=cell(1,3);
for i=1:1:3
    X1{i}=X{i}';
end
X=X1;
%Y=y;
% X=data;
% Y=truelabel{1};
num_view=length(X);
for j=1:num_view
    nor = mapstd(X{j},0,1); % Normalize X
    X{j} = nor;
end 
%Y=y;
% for i=[1e-5,1e-3,1e-1,1e1,1e3,1e5]
%     for j=[1e-5,1e-3,1e-1,1e1,1e3,1e5]
%for i=1.1:0.1:3.0
%for i=0.1:0.1:1
miu=1000;
rho=1.1;
maxmiu=1e6;
lambda1=1e5;
lambda2=1;
r=2.0;
p=0.6;
Iter_max =250;
tic
F=DMFC_SAGL(num_p,X,Y,miu,rho,maxmiu,lambda1,lambda2,r,p,Iter_max);
toc
[~, label] = max(F');
result = Clustering8Measure(label',Y); % 'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'
fprintf("--------------------miu:%d,rho:%f,lambda1:%d,lambda2:%d,r:%f,p:%f,num_p:%d-----------------------\n",miu,rho,lambda1,lambda2,r,p,num_p);
fprintf("ACC:%f,NMI:%f,PUR:%f,Fscore:%f\n",result(1)*100,result(2)*100,result(3)*100,result(4)*100)
%     end
%    % disp(['运行时间: ',num2str(toc)]);
% end
diary off