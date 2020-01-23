%% ������ѵ��ģ��
function [Model,Predict_Results,Accuracy]=SVM_Identification(Train,y)
% Input
% Train     ѵ����
% y         ���Լ�����ʵ���
% Output
% Model               ģ��
% Predict_Results     Ԥ����
% Accuracy            Ԥ��׼ȷ��
%
% Written By Gangying Lau
%
%% ����ģ�͡���SVM
Model=fitcsvm(Train(:,5:end),y,'Standardize',true,'KernelFunction','polynomial','KernelScale','auto');
% rbf
% polynomial
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