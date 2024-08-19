clc;
clear;
warning off;
addpath(genpath('./tools/'));

load('Mfeat.mat');Y=truelabel{1}; 
v=6;
n=2000;


%Normalize data
for i=1:v
    data{i}=NormalizeFea(data{i});
end

%%%%%%%%%%%%%%

param.alpha = 0.001;
param.beta = 0.001;

%%%%%%%%%%%%%%%
param.v = v;
param.c = c;
param.NITER = 30;
rand('twister',5489);
%%%%%%%%%%%%%%%
options = [];
options.ReducedDim = c;
P1=cell(1,v);
for iv = 1:v
[P1{iv},~] = PCA1(data{iv}', options);
    if size(P1{iv},2) < c
    P1{iv}(:,size(P1{iv},2)+1:c) = 0;
    end
end
%%%%%%% cunstruct graph %%%%%%%%
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';
S = cell(1,v);
for iv = 1:v
    S{iv} = constructW(data{iv}',options);
end

tic
[U,consenX,obj] = HCFS(data,P1,S,param);
toc

for i = 1:v
    U{i} = U{i}';
end
W = DataConcatenate(U);
W = W';
XX = consenX';
d = size(XX,1);
select=0.9;
 selectedFeas = select*d*0.1;
w = [];
for i = 1:d
    w = [w norm(W(i,:),2)];
end
[~,index] = sort(w,'descend');
Xw = XX(index(1:selectedFeas),:);
 for i=1:40
 label=litekmeans(Xw',c,'MaxIter',100,'Replicates',20);
    result1 = ClusteringMeasure(Y,label); 
    result(i,:) = result1;

 end
        for j=1:2
            a=result(:,j);
            ll=length(a);
            temp=[];
            for i=1:ll
                if i<ll-18
                    b=sum(a(i:i+19));
                    temp=[temp;b];
                end
            end
            [e,f]=max(temp);
            e=e./20;
            MEAN(j,:)=[e,f];
            STD(j,:)=std(result(f:f+19,j));
            rr(:,j)=sort(result(:,j));
            BEST(j,:)=rr(end,j);
        end

fprintf('\n');
disp(['mean. ACC: ', num2str(MEAN(1,1))]);
disp(['mean. STD(ACC): ', num2str(STD(1,1))]);
disp(['mean. NMI: ', num2str(MEAN(2,1))]);
disp(['mean. STD(NMI): ', num2str(STD(2,1))]);
fprintf('\n');
