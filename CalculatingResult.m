%% ������ʶ��ģ��
% Written BY Gangying Lau
tic
% ��������
% ������
%load('C:\Ruby\����������\����\���ݼ�ȫ����\������ʶ��\���ݼ����\ʶ��������\���������ݼ�_BY_FBY.mat')
% ������
load('C:\Ruby\����������\����\���ݼ�ȫ����\������ʶ��\���ݼ����\ʶ��������\���������ݼ�_BY_FBY.mat')
%% ����˵��
% ��� [�����䣻�Ǳ�����]
L_Tr=[];L_Te=[];
% Ԥ����
Tr_PR=[];Te_PR=[];
% ����ͳ��
A=[];B=[];
% ׼ȷ��
Tr_Ac=[];Te_Ac=[];
% �������ʶ��׼ȷ��
Co_BY=zeros(2,5);
% �Ǳ������ʶ��׼ȷ����
Co_FBY=zeros(2,5);
for i=1:5
% ��������
[Train, Test, N_Tr, N_Te]=Data_Sep(BY,FBY);
L_Tr=[L_Tr N_Tr];L_Te=[L_Te N_Te];
% ѵ��
[Model, R1, Accuracy]=SVM_Identification(Train,Train(:,3));
Tr_PR=[Tr_PR R1];
A1=tabulate(R1-Train(:,3));
A=[A; [i zeros(1,2)]; A1];
Tr_Ac=[Tr_Ac Accuracy];
% ����
[R2,Accuracy]=SVM_Identification_test2(Model,Test,Test(:,3));
Te_PR=[Te_PR R2];
B1=tabulate(R2-Test(:,3));
B=[B; [i zeros(1,2)]; B1];
Te_Ac=[Te_Ac Accuracy];
% ���������ͷǱ������Ԥ����ȷ��
Co_BY(1,i)=(1629-A1(find(A1(:,1)==-1),2))/1629;
Co_BY(2,i)=(181-B1(find(B1(:,1)==-1),2))/181;
Co_FBY(1,i)=(8145-A1(find(A1(:,1)==1),2))/8145;
Co_FBY(2,i)=(905-B1(find(B1(:,1)==1),2))/905;
end
toc