%function [accuracy,ytest0,predicted_class,train_Time]=Large_Scale_NonL_LSTWSVM(A,A_test,c,c0,mew,ir)
function [acc,obsX,Predict_Y,time,output_struct]=Large_Scale_NonL_LSTWSVM(A,A_test,FunPara)
%c=FunPara.c1;
%mew1=1/(2*mew*mew);
%[m,n]=size(A);[m_test,n_test]=size(A_test);
% x0=A(:,1:end-1);y0=A(:,end);
% %xtest0=A_test(:,1:n_test-1);ytest0=A_test(:,n_test);
% Cf=[x0 y0];
tic
c0=FunPara.c0;
ir=FunPara.ir;
C=nufuzz2(A,c0,ir);
clear A
%time1=toc;
%  C=nufuzz(Cf,c0);
%[no_input,no_col]=size(C);

Amem=C(C(:,end-1)==1,end);
Bmem = C(C(:,end-1)~=1,end);
C=C(:,1:end-1);
A=C(C(:,end)==1,1:end-1);
B=C(C(:,end)~=1,1:end-1);
clear C
% [no_input,no_col]=size(C);
% obs = C(:,end);
% A = [];
% B = [];
% Amem = [];
% Bmem = [];
% for i = 1:no_input
%     if(obs(i) == 1)
%         A = [A;C(i,1:end-1)];
%         Amem=[Amem;mem(i)];
%     else
%         B = [B;C(i,1:end-1)];
%         Bmem=[Bmem;mem(i)];
%     end
% end
%% Model-1
c1 = FunPara.c1;
c2 = c1;% c1=c2
c3 = FunPara.c3;%c3=c4
c4 = c3;
kerfPara = FunPara.kerfPara;
eps=10^-4;% For NDC eps=10^-2 and for small scale eps=10^-4
%eps=FunPara.eps_val;%% FOR SMO
p=size(A,1);
q=size(B,1);
v1=[zeros(1,p) ones(1,q)];
v2=[zeros(1,q) ones(1,p)];
Inv_S22=(1./Bmem);
Inv_S2=diag(Inv_S22);
IQ=(Inv_S2)'*(Inv_S2);
clear Inv_S22 Inv_S2 Bmem
Q=[kernelfun(A,kerfPara,A)+c3*eye(p),kernelfun(A,kerfPara,B);kernelfun(B,kerfPara,A),kernelfun(B,kerfPara,B)+(c3/c1)*IQ]+ones(p+q);
%Q=[kernelfun(A,kerfPara,A)+c3*eye(p),kernelfun(A,kerfPara,B);kernelfun(B,kerfPara,A),kernelfun(B,kerfPara,B)+c3*eye(q)/c1]+ones(p+q);
x=LSSMO(Q,eps,c3,v1);

clear  IQ Q v1 AAT ABT BAT BBT;

%x=bestx
l=size(x,1);
alpha=x(1:p,1);
beta=x(p+1:l,1);
%% Model-2
Inv_S11=(1./Amem);
Inv_S1=diag(Inv_S11);
IP=(Inv_S1)'*(Inv_S1);
clear Inv_S11 Inv_S1 Amem
H=[kernelfun(B,kerfPara,B)+c4*eye(q),kernelfun(B,kerfPara,A);kernelfun(A,kerfPara,B),kernelfun(A,kerfPara,A)+(c4/c2)*IP]+ones(p+q);
% E=ones(size(H1,1));
% H=H1+E;
% H=H1+ones(size(H1,1));
%H=(H+H')/2;
y=LSSMO(H,eps,c4,v2);
time=toc;
clear Inv_S11 Inv_S1 IP H v2;
%clear H1 H v2;
%y=besty
l=size(y,1);
lambda=y(1:q,1);
gamma=y(q+1:l,1);
% memory
% clear AA2;
% clear BB2;
%clear y;


%[no_test,m1]=size(TestX);

%if strcmp(kerfPara.type,'lin')
%    w1=(A'*alpha+B'*beta)/c3;
%    b1=(e2'*alpha+e1'*beta)/c3;
%    w2=(B'*lambda+A'*gamma)/c4;
%    b2=(e1'*lambda+e2'*gamma)/c4;
%    P_1=TestX(:,1:m1-1);
%    y1=(P_1*w1+b1);
%   y2=(P_1*w2+b2);
%else
% C=[A;B];
TestX=A_test;
P1=TestX(:,1:end-1);
obsX = TestX(:,end);
clear TestX
%P1=kernelfun(TestX(:,1:m1-1),kerfPara,TestX(:,1:m1-1));

%clear TestX;

e1=ones(q,1);
e2=ones(p,1);
b1=(e2'*alpha+e1'*beta)/c3;
b2=-(e1'*lambda+e2'*gamma)/c4;
y1=(kernelfun(P1,kerfPara,A)*alpha+kernelfun(P1,kerfPara,B)*beta)/c3+b1;
y2=-(kernelfun(P1,kerfPara,B)*lambda+kernelfun(P1,kerfPara,A)*gamma)/c4+b2;
%end
%clear P1 kernelfun(P1,kerfPara,A) kernelfun(P1,kerfPara,B) kernelfun(P1,kerfPara,B) kernelfun(P1,kerfPara,A);
%clear alpha beta lambda gamma A B e1 e2;
for i=1:size(y1,1)
    if (min(abs(y1(i)),abs(y2(i)))==abs(y1(i)))
        Predict_Y(i,1) = 1;
    else
        Predict_Y(i,1) =-1;
    end
    %dec_bdry(i,1)=min(abs(y1(i)),abs(y2(i)));
end
acc=length(find(obsX==Predict_Y))/length(Predict_Y);
acc=acc*100;
st = dbstack;
output_struct.function_name= st.name;
%% Previous
% %----------------Training-------------
% % A=x0(find(y0(:,1)>0),:);B=x0(find(y0(:,1)<=0),:);
% C=[A;B];
% m1=size(A,1);m2=size(B,1);m3=size(C,1);
% e1=ones(m1,1);e2=ones(m2,1);
% K = zeros (m1,m3);
% tic
% for i =1: m1
%     for j =1: m3
%         %         u=[A(i ,:)];v=[C(j ,:)];
%         nom = norm( A(i ,:) - C(j ,:) );
%         K(i,j)=exp(-mew1*nom*nom);
%         %         H1(i,j ) =exp(-mew*((u-v)*(u-v)'));
%     end
% end
% S1=diag(Amem);
% S2=diag(Bmem);
% G1=[K e1];
% G2=[S1*K S1*e1];
% GGT1=G1*G1';
% GGT2=G2*G2';
% % GTG=G'*G;
% % GGT=diag(1./Amem)*GGT;
% % invGTG=inv(GTG +(1e-5*speye(size(GTG,1))));
% K = zeros (m2,m3);
% for i =1: m2
%     for j =1: m3
%         %         u=[B(i ,:)];v=[C(j ,:)];
%         nom = norm( B(i ,:) - C(j ,:) );
%         K(i,j)=exp(-mew1*nom*nom);
%         %         H2(i,j ) = exp(-mew*((u-v)*(u-v)'));
%     end
% end
% H1=[S2*K S2*e2];
% H2=[K e2];
% HHT1=H1*H1';
% HHT2=H2*H2';
% % HTH=H'*H;
% % HHT=diag(1./Bmem)*HHT;
% % u1=0;u2=0;
% eps=1e-5;
% I1=speye(m+1);
% % S=[];
%
% if (m1<m2)
%     I2=speye(m2);
%     Y1=eps\(I1-H1'*((eps*I2+HHT1)\H1));
%     GY1=G1*Y1;YGT1=Y1*G1';GYGT1=G1*YGT1;
%     Y2=eps\(I1-H2'*((eps*I2+HHT2)\H2));
%     GY2=G2*Y2;YGT2=Y2*G2';GYGT2=G2*YGT2;
%     I=speye(m1);
%     u1=-(Y1-YGT1*((c*I+GYGT1)\GY1))*H1'*S2*e2;
%     u2=c*(Y2-YGT2*(((c\I)+GYGT2)\GY2))*G2'*e1;
% else
%     I2=speye(m1);
%     Z1=(I1-G1'*((eps*I2+GGT1)\G1))*(1/eps);
%     HZ1=H1*Z1;ZHT1=Z1*H1';HZHT1=H1*ZHT1;
%     Z2=(I1-G2'*((eps*I2+GGT2)\G2))*(1/eps);
%     HZ2=H2*Z2;ZHT2=Z2*H2';HZHT2=H2*ZHT2;
%     I=speye(m2);
%     u1=-c*(Z1-ZHT1*((c\I+HZHT1)\HZ1))*H1'*S2*e2;
%     u2=(Z2-ZHT2*(((c*I)+HZHT2)\HZ2))*G2'*e1;
% end
% % mem1=ones(size(mem,1),1)-mem;
% % u1=-inv(HTH+(diag(1./(c*[mem; 1])).*GTG)+(1e-5*speye(size(HTH,1))))*H'*e2;
% % u2=inv(GTG+(diag(1./(c*[mem; 1])).*HTH)+(1e-5*speye(size(GTG,1))))*G'*e1;
%
% % u1=-inv(HTH+(1/c*GTG)+(1e-5*speye(size(HTH,1))))*H'*e2;
% % u2=inv(GTG+(1/c*HTH)+(1e-5*speye(size(GTG,1))))*G'*e1;
% train_Time=time1+toc;
% %---------------Testing---------------
%
% no_test=size(xtest0,1);
% K = zeros(no_test,m3);
% for i =1: no_test
%     for j =1: m3
%         %         u=xtest0(i ,:);v=C(j ,:);
%         nom = norm( xtest0(i ,:) - C(j ,:) );
%         K(i,j )=exp(-mew1*nom*nom);
%     end
% end
% K=[K ones(no_test,1)];
% preY1=K*u1/norm(u1(1:size(u1,1)-1,:));preY2=K*u2/norm(u2(1:size(u2,1)-1,:));
% predicted_class=[];
% for i=1:no_test
%     if abs(preY1(i))< abs(preY2(i))
%         predicted_class=[predicted_class;1];
%     else
%         predicted_class=[predicted_class;-1];
%     end
%
% end
% % err = sum(predicted_class ~= ytest0);
% % accuracy=(no_test-err)/(no_test)*100;
%
% %%%%%%%Imbalance accuracy
% no_test=m_test;
% classifier=predicted_class;
% obs1=ytest0;
% match = 0.;
% match1=0;
% % classifier = classifier';
% % obs1 = test_data(:,no_col);
% posval=0;
% negval=0;
% %[test_size,n] = size(classifier);
% for i = 1:no_test
%     if(obs1(i)==1)
%         if(classifier(i) == obs1(i))
%             match = match+1;
%         end
%         posval=posval+1;
%     elseif(obs1(i)==-1)
%         if(classifier(i) ~= obs1(i))
%             match1 = match1+1;
%         end
%         negval=negval+1;
%     end
% end
% if(posval~=0)
%     a_pos=(match/posval)
% else
%     a_pos=0;
% end
%
% if(negval~=0)
%     am_neg=(match1/negval)
% else
%     am_neg=0;
% end
%
% AUC=(1+a_pos-am_neg)/2;
%
% accuracy=AUC*100
%

%return
end
