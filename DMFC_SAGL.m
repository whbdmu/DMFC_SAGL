function [F_sum] = DMFC_SAGL(num_p,X,Y,miu,rho,maxmiu,lambda1,lambda2,r,p,Iter_max)
%MAINMETHOD 此处显示有关此函数的摘要
%   此处显示详细说明
    num_view=length(X);
    num_sample = size(X{1},2);
    num_clusters=max(Y);
    alpha = ones(num_p,1)/num_p;
    A = cell(num_p,num_view);
    Z = cell(num_p,1);
    M = cell(num_p,1);
    F = cell(num_p,1);
    G = cell(num_p,1);
    Y1 = cell(num_p,1);
    Y2 = cell(num_p,1);
    Q = cell(num_p,1);
    K = cell(num_p,1);
    obj=zeros(1,Iter_max);
    for i=1:num_p
        Z{i}=zeros(num_clusters*i,num_sample);
        F{i}=zeros(num_sample,num_clusters);
        Y1{i}=F{i};
        Y2{i}=F{i};
        for j=1:num_clusters
            F{i}(j,j)=1;
        end
        Q{i}=F{i};
        K{i}=F{i};
        M{i}=zeros(num_clusters*i,num_clusters);
        for j=1:num_view
            % initialize A      
            rand('twister',5489);
            [~, anc] = litekmeans(X{j}',num_clusters*i,'MaxIter', 100,'Replicates',10);
            A{i,j} = anc';
        end 
    end
    iter=1;
%% ================= Iterative Update ===========================
    while iter<=Iter_max
        %%=================Zp=================================
        for i=1:num_p
            C=0;
            for j=1:num_view
                C=C+A{i,j}'*X{j};
            end
            Z{i}=(C+(M{i}*F{i}'))./(num_view+1);
        end
        %%=================Ap(v)=================================
        for i=1:num_p
            for j=1:num_view
                B=X{j}*Z{i}';
                [U,~,V] = svd(B,'econ');
                A{i,j}=U*V';
            end
        end
        %%=================Mp=================================
        for i=1:num_p
            M{i}=Z{i}*F{i};
        end
     
        %%=================Q=================================
        for i =1:num_p
            G{i} =F{i} + Y1{i}./miu;
        end
        G_tensor = cat(3,G{:,:});
        H = G_tensor(:);
        sX=[num_sample,num_clusters, num_p];
        [myq, ~] = wshrinkObj_weight_lp(H,ones(1, num_p)'.*(1*lambda1/miu),sX, 0,3,p);
        Q_tensor = reshape(myq, sX);
        for i=1:num_p
            Q{i} = Q_tensor(:,:,i);
        end
       %%================= Kp=================================
        for i=1:num_p
            K{i}=F{i}+(Y2{i}./miu);
            K{i}(K{i}<0)=0;
        end     
        %%=================Fp=================================
%         for i=1:num_p
%             D=0;
%             for j=1:num_p
%                 if j~=i
%                     D=D+F{j};
%                 end
%             end
%             B=2*(alpha(i)^r)*Z{i}'*M{i}-lambda2*D+miu*Q{i}-Y1{i}+miu*K{i}-Y2{i};
%             [U,~,V] = svd(B,'econ');
%             F{i}=U*V';
%         end
        D=0;
        for i=1:num_p
            D=D+F{i};
        end
        for i=1:num_p
            B=2*(alpha(i)^r)*Z{i}'*M{i}-lambda2*(D-F{i})+miu*Q{i}-Y1{i}+miu*K{i}-Y2{i};
            [U,~,V] = svd(B,'econ');
            F{i}=U*V';
        end
        %%================= alpha(p)=================================
        n=zeros(1,num_p);
        m=zeros(1,num_p);
        n_sum=0;
        for i=1:num_p
            for j=1:num_view
                n(i)=n(i)+(norm(X{j}-A{i,j}*Z{i},'fro')^2);
            end
            m(i)=n(i)+(norm(Z{i}-M{i}*F{i}','fro')^2);
            n(i)=(m(i))^(1/(1-r));
            n_sum=n_sum+n(i);
        end
        alpha=n/n_sum;
        %%================= miu,Y1,Y2=================================
        miu = min(rho*miu, maxmiu);
        for i=1:num_p
            Y1{i}=Y1{i}+miu*(F{i}-Q{i});
            Y2{i}=Y2{i}+miu*(F{i}-K{i});
        end

        %%==================convergence=================================
        RR1 = [];RR2=[];
        for i=1:num_p
            res1=F{i}-Q{i};
            res2=F{i}-K{i};
            RR1 =[RR1, norm(res1,'inf')];
            RR2 =[RR2, norm(res2,'inf')];
        end
        error1=norm(RR1, inf);
        error2=norm(RR2, inf);
        fprintf("iter:%d,F-Q:%f,F-K:%f\n",iter,error1,error2);
        if(error1<1e-5 && error2<1e-5)
            break;
        end
        iter=iter+1;
    end
    F_sum = zeros(num_sample, num_clusters);
    for i=1:num_p
        F_sum = F_sum + alpha(i)^r*F{i};                                  
    end
end

