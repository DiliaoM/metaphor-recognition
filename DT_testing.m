tic
% ��������
% ������
%load('���������ݼ�_BY_FBY.mat')
% ������
load('���������ݼ�_BY_FBY.mat')
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
Num=randperm(size(BY,1));
L_Tr=[L_Tr Num(184:end)'];
L_Te=[L_Te Num(1:183)'];
% ѵ�����Ͳ��Լ�
Train=BY(Num(184:end),:);
Test=BY(Num(1:183),:);
% ѵ��
[Model, R1, Accuracy]=Tr_Tree(Train,Train(:,4));
Tr_PR=[Tr_PR R1];
A1=tabulate(R1-Train(:,4));
A=[A; [i zeros(1,2)]; A1];
Tr_Ac=[Tr_Ac Accuracy];
% ����
[R2,Accuracy]=Te_Tree(Model,Test,Test(:,4));
Te_PR=[Te_PR R2];
B1=tabulate(R2-Test(:,4));
B=[B; [i zeros(1,2)]; B1];
Te_Ac=[Te_Ac Accuracy];
% ���������ͷǱ������Ԥ����ȷ��
a=size(find(Train(:,4)==1),1);
Co_BY(1,i)=(a-A1(find(A1(:,1)==-1),2))/a;
c=size(find(Test(:,4)==1),1);
Co_BY(2,i)=(c-B1(find(B1(:,1)==-1),2))/c;

b=size(find(Train(:,4)==0),1);
Co_FBY(1,i)=(b-A1(find(A1(:,1)==1),2))/b;
d=size(find(Test(:,4)==0),1);
Co_FBY(2,i)=(d-B1(find(B1(:,1)==1),2))/d;
end
toc