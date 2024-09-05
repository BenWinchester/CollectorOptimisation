clc;clear;



SR_Si=xlsread('data', 'solar cells', 'I3:I2004'); % Si, SunPowe C60 Solar Cell,cell efficiency 25.1%
SR_CdTe=xlsread('data', 'solar cells', 'D3:D2004'); % CdTe, First Solar 4 sereries,cell efficiency 22.1%

AM=xlsread('data', 'ambient conditions', 'C2:C2003'); %the intensity distribution of AM1.5
GAM=1000.37;%The intensity of the standard AM1.5

AM_wl=xlsread('data', 'ambient conditions', 'B2:B2004'); %the wavelength distribution of AM1.5

Ac=1;%the area

for j=1:73

for k=1:123
    
%define the filter parameters
cutl=280+(j-1)*10;%the lower cut of the filter,nm
cuth=cutl+(k-1)*10;%the higher cut of the filter,nm

%filter:the transparence at cutl~cuth is 1, otherwise zero.
for i=1:2002
    if cutl<AM_wl(i)&&AM_wl(i)<cuth
        Trfl(i)=1;
        Abfl(i)=0;
        Refl(i)=0;
    else
        Trfl(i)=0;%transmisivity of the filter
        Abfl(i)=1;
        Refl(i)=0;
    end
end

[Gpv,Grel,Gabs]=Gfilter(AM,AM_wl,Trfl,Refl,Abfl);

Si=solarcell(Ac,Gpv,GAM,SR_Si,SR_CdTe,AM_wl,'silicon');
CdTe=solarcell(Ac,Gpv,GAM,SR_Si,SR_CdTe,AM_wl,'CdTe');

cutL(j,k)=cutl;
cutH(j,k)=cuth;

eff_el_Si(j,k)=Si(1);%electrical efficieny
eff_el_CdTe(j,k)=CdTe(1);%electrical efficieny
eff_el_diff(j,k)=Si(1)-CdTe(1);%electricity efficiency difference

eff_wth_Si(j,k)=(GAM-Gabs)/GAM-Si(1);%waste heat in the PV
eff_wth_CdTe(j,k)=(GAM-Gabs)/GAM-CdTe(1);%waste heat in the PV

eff_th(j,k)=Gabs/GAM;%thermal efficiency

eff_total_Si(j,k)=Si(1)+Gabs/GAM;%total efficiency
eff_total_CdTe(j,k)=CdTe(1)+Gabs/GAM;%total efficiency

end

end