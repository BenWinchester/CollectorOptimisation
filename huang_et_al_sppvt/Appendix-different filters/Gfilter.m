%calculate the passed, reflected and absorbed solar by a filter
function [Gpass,Grel,Gabs]=Gfilter(Gin,lamadaG,Tr,Rel,Abs)

N=size(Gin);
Gabs=0;
%lamadaG=xlsread('data', 'ambient conditions', 'B2:B2004');
Gpass=zeros(N(1),1);
Grel=zeros(N(1),1);

for i=1:N(1)
    Gpass(i)=Gin(i)*Tr(i);
    Grel(i)=Gin(i)*Rel(i);
    Gabs=Gabs+Gin(i)*Abs(i)*(lamadaG(i+1)-lamadaG(i));
end

end