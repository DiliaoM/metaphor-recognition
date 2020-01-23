%% ʶ�������ȡ����
function [Train, Test, L_Tr, L_Te]=Data_Sep(BY,FBY)
% Input
% BY      ������
% FBY     �Ǳ�����
% Output
% Train   ѵ����
% Test    ���Լ�
% L_Tr    ѵ��������Դ���
% L_Te    ����������Դ���
%
% Written By Gangying Lau
%
%% �������ʮ�۱��
Num1=randperm(size(BY,1));
Num_BY=reshape(Num1(1:(size(BY,1)-mod(size(BY,1),10))),[10 (size(BY,1)-mod(size(BY,1),10))/10]);
%% �Ǳ������ʮ�۱��
Num2=randperm(size(FBY,1));
Num_FBY=reshape(Num2(1:(size(FBY,1)-mod(size(FBY,1),10))),[10 (size(FBY,1)-mod(size(FBY,1),10))/10]);
%% ��װ���ݼ�
% ����
[a,b]=size(Num_BY(1:9,:));
A=reshape(Num_BY(1:9,:),[1 a*b]);
% �Ǳ���
Num_FBY=Num_FBY(1:10,1:5*b);
[c,d]=size(Num_FBY(1:9,:));
B=reshape(Num_FBY(1:9,:),[1 c*d]);
% ѵ��������װ
Train=[BY(A,:); FBY(B,:)];
Test=[BY(Num_BY(10,:),:); FBY(Num_FBY(10,:),:)];
L_Tr=[A'; B'];
L_Te=[Num_BY(10,:)'; Num_FBY(10,:)'];