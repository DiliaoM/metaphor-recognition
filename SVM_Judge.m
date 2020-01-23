%% 比喻句识别
function [Model,Predict_Results,Accuracy]=Tr_Tree(Train,y)
% 建立模型——SVM
Model=fitctree(Train(:,5:end),y);
%
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