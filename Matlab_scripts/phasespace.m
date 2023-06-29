% by Alexandros Leontitsis
function [T , T1]=phasespace(x,dim,tau)

N=length(x);
T1=N-(dim-1)*tau;
T=zeros(T1,dim);

for i=1:T1
   T(i,:)=x(i+(0:dim-1)*tau)';
end
% T=T';