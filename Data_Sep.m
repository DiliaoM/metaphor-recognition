%% 识别随机抽取数据
function [Train, Test, L_Tr, L_Te]=Data_Sep(BY,FBY)
% Input
% BY      比喻句
% FBY     非比喻句
% Output
% Train   训练集
% Test    测试集
% L_Tr    训练数据来源编号
% L_Te    测试数据来源编号
%
% Written By Gangying Lau
%
%% 比喻句的十折编号
Num1=randperm(size(BY,1));
Num_BY=reshape(Num1(1:(size(BY,1)-mod(size(BY,1),10))),[10 (size(BY,1)-mod(size(BY,1),10))/10]);
%% 非比喻句的十折编号
Num2=randperm(size(FBY,1));
Num_FBY=reshape(Num2(1:(size(FBY,1)-mod(size(FBY,1),10))),[10 (size(FBY,1)-mod(size(FBY,1),10))/10]);
%% 组装数据集
% 比喻
[a,b]=size(Num_BY(1:9,:));
A=reshape(Num_BY(1:9,:),[1 a*b]);
% 非比喻
Num_FBY=Num_FBY(1:10,1:5*b);
[c,d]=size(Num_FBY(1:9,:));
B=reshape(Num_FBY(1:9,:),[1 c*d]);
% 训练集的组装
Train=[BY(A,:); FBY(B,:)];
Test=[BY(Num_BY(10,:),:); FBY(Num_FBY(10,:),:)];
L_Tr=[A'; B'];
L_Te=[Num_BY(10,:)'; Num_FBY(10,:)'];