function [obj] = cal_tensorSp(X,p)
%CAL_TENSORSP 此处显示有关此函数的摘要
% input X
% output ||X`||_Sp^p; 

%X`is mode3，shiftdim(X, 1);
%   此处显示详细说明
for v = 1:length(X)
    XX(v,:,:) = X{v};
end
Y=shiftdim(XX, 1);
Yhat = fft(Y,[],3);
obj = 0;
for v = 1: size(X,1)
    [~,shat,~] = svd(full(Yhat(:,:,v)),'econ');
    shat=diag(shat);
    obj = obj + sum(shat.^p);
end
end

