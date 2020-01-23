%% 比喻句识别模型
% Written BY Gangying Lau
tic
% 导入数据
% 词向量
%load('C:\Ruby\大三下资料\课题\数据集全填完\比喻句识别\数据集拆分\识别与评价\词向量数据集_BY_FBY.mat')
% 句向量
load('C:\Ruby\大三下资料\课题\数据集全填完\比喻句识别\数据集拆分\识别与评价\句向量数据集_BY_FBY.mat')
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
% 生成数据
[Train, Test, N_Tr, N_Te]=Data_Sep(BY,FBY);
L_Tr=[L_Tr N_Tr];L_Te=[L_Te N_Te];
% 训练
[Model, R1, Accuracy]=SVM_Identification(Train,Train(:,3));
Tr_PR=[Tr_PR R1];
A1=tabulate(R1-Train(:,3));
A=[A; [i zeros(1,2)]; A1];
Tr_Ac=[Tr_Ac Accuracy];
% 测试
[R2,Accuracy]=SVM_Identification_test2(Model,Test,Test(:,3));
Te_PR=[Te_PR R2];
B1=tabulate(R2-Test(:,3));
B=[B; [i zeros(1,2)]; B1];
Te_Ac=[Te_Ac Accuracy];
% 计算比喻句和非比喻句的预测正确率
Co_BY(1,i)=(1629-A1(find(A1(:,1)==-1),2))/1629;
Co_BY(2,i)=(181-B1(find(B1(:,1)==-1),2))/181;
Co_FBY(1,i)=(8145-A1(find(A1(:,1)==1),2))/8145;
Co_FBY(2,i)=(905-B1(find(B1(:,1)==1),2))/905;
end
toc