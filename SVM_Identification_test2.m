%% 比喻句模型测试
function [Predict_Results,Accuracy]=SVM_Identification_test2(Model,Test,y)
% Input
% Model     SVM模型
% Test      测试集
% y         测试集的真实结果
% Output
% Predict_Results     预测结果
% Accuracy            预测准确率
%
% Written By Gangying Lau
%
%% 测试
Predict_Results=predict(Model,Test(:,5:end));
% 计算分类的正确率
Correct_Results=Predict_Results-y;
Results=tabulate(Correct_Results);
[m,n]=find(Results(:,1)==0);
Accuracy= Results(m,3);
% 输出两个数字的测试分类正确率
fprintf('The testing accuracy from Matlab decision tree function is %f%%\n', Accuracy);