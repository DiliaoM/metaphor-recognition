%% 数据测试集创建
function [Predict_Results,Accuracy]=Te_Tree(Model,Test,y)
%% 测试
Predict_Results=predict(Model,Test(:,5:end));
% 计算分类的正确率
Correct_Results=Predict_Results-y;
Results=tabulate(Correct_Results);
[m,n]=find(Results(:,1)==0);
Accuracy= Results(m,3);
% 输出两个数字的测试分类正确率
fprintf('The testing accuracy from Matlab decision tree function is %f%%\n', Accuracy);