clear all
clc
lamdashu=12;
%��ȡ���������е�pssm
WEISHU=369;
load('RPI369_protein_N_lensec.mat')
for i=1:WEISHU
    nnn=num2str(i);
    name = strcat(nnn,'.pssm');
    fid{i}=importdata(name);
end
%���е��������й�һ��
c=cell(WEISHU,1);
for t=1:WEISHU
    clear shu d
shu=fid{t}.data;
%֪��ÿ�������ʵ���������ȡ�����ľ���ע�⵰���ʵ�˳��
% shuju=shu(1:i,1:20);
[M,N]=size(shu);
shuju=shu(1:RPI369_protein_N_lensec(1,t),1:20);
d=[];
%��һ��
for i=1:RPI369_protein_N_lensec(1,t)
   for j=1:20
       d(i,j)=1/(1+exp(-shuju(i,j)));
   end
end
c{t}=d(:,:);
end
%����PSSM-AAC,x��һ��,
for i=1:WEISHU
[MM,NN]=size(c{i});
 for  j=1:20
   x(i,j)=sum(c{i}(:,j))/MM;
 end
end
%PsePSSM��20*lamda
xx=[];
sheta=[];
shetaxin=[];
% lamda=1;
for lamda=1:lamdashu;
for t=1:WEISHU
  [MM,NN]=size(c{t});
  clear xx
   for  j=1:20
      for i=1:MM-lamda
       xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
      end
      sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
   end
end
shetaxin=[shetaxin,sheta];
end
psepssm=[x,shetaxin];
save psepssmM.mat psepssm
      