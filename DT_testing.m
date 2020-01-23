tic
% 导入数据
% 词向量
%load('词向量数据集_BY_FBY.mat')
% 句向量
load('句向量数据集_BY_FBY.mat')
%% 变量说明
% 编号 [比喻句；非比喻句]
L_Tr=[];L_Te=[];
% 预测结果
Tr_PR=[];Te_PR=[];
% 错误统计
A=[];B=[];
% 准确率
Tr_Ac=[];Te_Ac=[];
% 比喻句的识别准确率
Co_BY=zeros(2,5);
% 非比喻句的识别准确利率
Co_FBY=zeros(2,5);
for i=1:5
Num=randperm(size(BY,1));
L_Tr=[L_Tr Num(184:end)'];
L_Te=[L_Te Num(1:183)'];
% 训练集和测试集
Train=BY(Num(184:end),:);
Test=BY(Num(1:183),:);
% 训练
[Model, R1, Accuracy]=Tr_Tree(Train,Train(:,4));
Tr_PR=[Tr_PR R1];
A1=tabulate(R1-Train(:,4));
A=[A; [i zeros(1,2)]; A1];
Tr_Ac=[Tr_Ac Accuracy];
% 测试
[R2,Accuracy]=Te_Tree(Model,Test,Test(:,4));
Te_PR=[Te_PR R2];
B1=tabulate(R2-Test(:,4));
B=[B; [i zeros(1,2)]; B1];
Te_Ac=[Te_Ac Accuracy];
% 计算比喻句和非比喻句的预测正确率
a=size(find(Train(:,4)==1),1);
Co_BY(1,i)=(a-A1(find(A1(:,1)==-1),2))/a;
c=size(find(Test(:,4)==1),1);
Co_BY(2,i)=(c-B1(find(B1(:,1)==-1),2))/c;

b=size(find(Train(:,4)==0),1);
Co_FBY(1,i)=(b-A1(find(A1(:,1)==1),2))/b;
d=size(find(Test(:,4)==0),1);
Co_FBY(2,i)=(d-B1(find(B1(:,1)==1),2))/d;
end
toc