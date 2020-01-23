%% ������ʶ��
function [Model,Predict_Results,Accuracy]=Tr_Tree(Train,y)
% ����ģ�͡���SVM
Model=fitctree(Train(:,5:end),y);
%
% �����ģ�͵Ĵ��������
tmp=crossval(Model);
classLoss=kfoldLoss(tmp)
% ʹ�ý�����֧�����������ж��������
Predict_Results=predict(Model,Train(:,5:end));
% ����������ȷ��
Correct_Results=Predict_Results-y;
Results=tabulate(Correct_Results);
[m,n]=find(Results(:,1)==0);
Accuracy= Results(m,3);
% ����������ֵĲ��Է�����ȷ��
fprintf('The training accuracy from Matlab decision tree function is %f%%\n', Accuracy);