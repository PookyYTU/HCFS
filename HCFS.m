function [Q, ConsenX,obj] = HCFS(X,P1,S,param)
%%paramter:

%% ===================== Parameters =====================
NITER = param.NITER;
v = param.v;
alpha=param.alpha;
beta=param.beta;
mu = 0.1;rho = 1.01;max_mu = 10^5;
wv = ones(1,v) ./ v;
%% ===================== initialize =====================
Z = cell(1,v);
Q = cell(1,v);
P = cell(1,v);
C = cell(1,v);
T = cell(1,v);
nFea=zeros(1,v);
Xcon=cell(1,v);
for vIndex=1:v
    [d,n] = size(X{vIndex});
    nFea(1,vIndex)=size(X{1,vIndex},1);
    Z{vIndex} = S{vIndex};
    Q{vIndex} = P1{vIndex};
    C{vIndex} = zeros(n,n);
    T{vIndex} = zeros(n,n);
end

%%%% Initialize L %%%%
S0 = zeros(n,n);
for iterv = 1:v
    S0 = S0 + S{iterv};
end
S0 = S0/v;
for j = 1:n
    d_sum = sum(S0(j,:));
    if d_sum == 0
        d_sum = eps;
    end
    S0(j,:) = S0(j,:)/d_sum;
end
S0 = (S0+S0')/2;
D = diag(sum(S0));
L = D - S0;

%%%% Initialize G %%%%
gj = cell(1,v);
g = cell(1,v);
G = cell(1,v);
for iterv=1:v
    g{iterv}=ones(nFea(iterv),1);
    G{iterv}=spdiags(g{iterv},0,nFea(iterv),nFea(iterv));
    G{iterv}=full(G{iterv});
end

for i = 1:v
    Xcon{i} = X{i}';
end
ConsenX = DataConcatenate(Xcon);

%% ===================== updating =====================
for iterv = 1:v
    P{iterv} = P1{iterv};
end
for iter = 1:NITER
 %% update Q
     for iterv = 1:v 
        Q{iterv} = (X{iterv}*Z{iterv}*Z{iterv}'*X{iterv}'+alpha*G{iterv}+beta*X{iterv}*L*X{iterv}')\(X{iterv}*Z{iterv}*X{iterv}'*P{iterv});
     end
 %% construct G
     for iterv = 1:v
        gj{iterv} = sqrt(sum(Q{iterv}.*Q{iterv},2)+eps);
        g{iterv} = 0.5./gj{iterv};
        G{iterv} = diag(g{iterv});
     end
 %% update Z
      for iterv = 1:v
        Z{iterv} = (X{iterv}'*Q{iterv}*Q{iterv}'*X{iterv}+mu*eye(n))\(X{iterv}'*Q{iterv}*P{iterv}'*X{iterv}+mu*T{iterv}-C{iterv});
      end
 %% update A
    tempZ = zeros(n, n);
      for iterv = 1:v
        tempZ = tempZ + wv(iterv) * Z{iterv};
      end
    tempZ = tempZ ./ sum(wv);
    A = tempZ - diag(diag(tempZ));
    A = max(0.5 * (A + A'), 0);
 %% update wv
    for iterv = 1 : v
        wv(iterv) = 0.5 / (norm(Z{iterv} - A, 'fro') + eps);
    end
 %% update L
      if iter == 1
            Weight = constructW_PKN(A, 15);
            Diag_tmp = diag(sum(Weight));
            L = Diag_tmp - Weight;
      else
            param.num_view = 15; 
            HG = gsp_nn_hypergraph(A, param);
            L = HG.L;
      end
 %% update tensor
    Z_tensor = cat(3, Z{:,:});
    C_tensor = cat(3, C{:,:});
    Zv = Z_tensor(:);
    Cv = C_tensor(:);
    [Tv, objV] = wshrinkObj(Zv + 1/mu*Cv,1/mu,[n,n,v],0,1);
    T_tensor = reshape(Tv, [n,n,v]);
    % -----------------------------------%
    for iv = 1:v
        T{iv} = T_tensor(:,:,iv);
    end
 %% update C
    for iv = 1:v
        C{iv} = C{iv}+mu*(Z{iv}-T{iv});
    end
     clear Z_tensor T_tensor Zv Tv Tv T_tensor
 %% update P
       for iterv = 1:v
            M = X{iterv}*Z{iterv}'*X{iterv}'*Q{iterv};
            [U1,~,S1] = svd(M,'econ');
            P{iterv} = U1*S1';   
       end
 %% update mu
        mu = min(rho*mu,max_mu);
%% ===================== calculate obj =====================
    tempobj=0;
    for objIndex=1:v
        Term1 = norm(X{objIndex} - P{objIndex}*Q{objIndex}'*X{objIndex}*Z{objIndex},'fro')^2;
        Term2 = sum(sqrt(sum(Q{objIndex}.*Q{objIndex},2)));
        Term3 = trace(Q{objIndex}'*X{objIndex}*L*X{objIndex}'*Q{objIndex});
        Term4 = norm(Z{objIndex} - A,'fro')^2;
        tempobj=tempobj+Term1+alpha*Term2+beta*Term3+wv(objIndex)*Term4;
    end
    obj(iter)=tempobj;  
    if iter == 1
        err = 0;
    else
        err = obj(iter)-obj(iter-1);
    end

    fprintf('iteration =  %d:  obj: %.4f; err: %.4f  \n', ...
        iter, obj(iter), err);
    if (abs(err))<1e+0
        if iter > 15
            break;
        end
    end
end










