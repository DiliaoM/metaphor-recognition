%% ���ݲ��Լ�����
function [Predict_Results,Accuracy]=Te_Tree(Model,Test,y)
%% ����
Predict_Results=predict(Model,Test(:,5:end));
% ����������ȷ��
Correct_Results=Predict_Results-y;
Results=tabulate(Correct_Results);
[m,n]=find(Results(:,1)==0);
Accuracy= Results(m,3);
% ����������ֵĲ��Է�����ȷ��
fprintf('The testing accuracy from Matlab decision tree function is %f%%\n', Accuracy);