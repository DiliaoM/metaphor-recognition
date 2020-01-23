%% 比喻句训练模型
function [Model,Predict_Results,Accuracy]=SVM_Identification(Train,y)
% Input
% Train     训练集
% y         测试集的真实结果
% Output
% Model               模型
% Predict_Results     预测结果
% Accuracy            预测准确率
%
% Written By Gangying Lau
%
%% 建立模型——SVM
Model=fitcsvm(Train(:,5:end),y,'Standardize',true,'KernelFunction','polynomial','KernelScale','auto');
% rbf
% polynomial
% 计算该模型的错误分类率
tmp=crossval(Model);
classLoss=kfoldLoss(tmp)
% 使用建立的支持向量机进行二分类测试
Predict_Results=predict(Model,Train(:,5:end));
% 计算分类的正确率
Correct_Results=Predict_Results-y;
Results=tabulate(Correct_Results);
[m,n]=find(Results(:,1)==0);
Accuracy= Results(m,3);
% 输出两个数字的测试分类正确率
fprintf('The training accuracy from Matlab decision tree function is %f%%\n', Accuracy);