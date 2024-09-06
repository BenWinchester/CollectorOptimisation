% %Figure1: spectrum distributions
figure;
x0=0;
y0=0;
width=560*1.5;
height=420*1.5;
set(gcf,'position',[x0,y0,width,height]);

x_G=readtable('data', sheet='ambient conditions', range='B2:B2003');
SR_CdTe=readtable('data', sheet='solar cells', range='D3:D2004');
SR_Sil=readtable('data', sheet='solar cells', range='I3:I2004');

Plotspectr(1)=plot(x_G,G0,'-','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Original Spectrum}');hold on;
Plotspectr(2)=plot(x_G,G4pass,'-','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Filtered Spectrum}');hold on;
Plotspectr(3)=plot(x_G,SR_CdTe,'--','Color',[1 0 0],'linewidth',2,'DisplayName','{CdTe Spectral Response}');
%Plotspectr(3)=plot(x_G,SR_Sil,'--','Color',[0 0 0],'linewidth',2,'DisplayName','{Silicon Spectral Response}');

axis([0 2500 0 1.8]);
xlabel('{Wavelength} [nm]','FontSize',16);
ylabel('{Spectral intensity [W/m2/nm]   Spectral Response [W/A]}','FontSize',14);
legenspectr=legend(Plotspectr,'Location','northeast');legend boxon;
set(legenspectr,'FontSize',18);
set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:500:2500],'YTick',[0:0.2:1.8],'TickDir','in','box','off');
set(gcf,'color','white')
box on;





%figure 2 efficiencies

figure;
x0=500;
y0=0;
width=560*1.5;
height=420*1.5;
set(gcf,'position',[x0,y0,width,height]);


Ploteff(1)=plot(Tr,effth,'-o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{\it\eta}_{t-Spectral splitting PVT}');hold on;
Ploteff(2)=plot(Tr1,effth1,'--o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{\it\eta}_{t-Conventional PVT}');hold on;
Ploteff(3)=plot(Tr,effthfl,'-','Color',[0 0 0],'linewidth',2,'DisplayName','{\it\eta}_{t-filter}');hold on;
%Plotspectr(3)=plot(Tr,effthcool,'-o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{\it\eta}_{t-coolant}');hold on;
Ploteff(4)=plot(Tr,effel,'-o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{\it\eta}_{el-Spectral splitting PVT}');
Ploteff(5)=plot(Tr1,effel1,'--o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{\it\eta}_{el-Conventional PVT}');
grid on;

axis([0 0.06 0 1]);
xlabel('{T}_{reduced} [-]','FontSize',16);
ylabel('{Efficiency [-]}','FontSize',16);
legendeff=legend(Ploteff,'Location','northeast');legend boxon;
set(legendeff,'FontSize',12);
set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:0.01:0.06],'YTick',[0:0.1:1],'TickDir','in','box','off');
set(gcf,'color','white')
box on;


%figure 3: energy balance of the filter

figure;
x0=1000;
y0=0;
width=560*1.5;
height=420*1.5;
set(gcf,'position',[x0,y0,width,height]);


Plotenergy(1)=plot(Tr,Energy(:,3),'-','Color',[0 0 0],'linewidth',2,'DisplayName','{Q}_{absorb}');hold on;
Plotenergy(2)=plot(Tr,Energy(:,4),'--o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Q}_{loss-radiation-top}');hold on;
Plotenergy(3)=plot(Tr,Energy(:,5),'-o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Q}_{loss-covection-top}');hold on;
Plotenergy(4)=plot(Tr,Energy(:,6),'--o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Q}_{loss-radiation-bottom}');hold on;
Plotenergy(5)=plot(Tr,Energy(:,7),'-o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Q}_{loss-convection-bottom}');hold on;
Plotenergy(6)=plot(Tr,Energy(:,8),'--','Color',[0 0 0],'linewidth',2,'DisplayName','{Q}_{balance}');

grid on;

axis([0 0.06 -0.7 0.5]);
xlabel('{T}_{reduced} [-]','FontSize',16);
ylabel('{Energy flow [-]}','FontSize',16);
legendenergy=legend(Plotenergy,'Location','southwest');legend boxon;
set(legendenergy,'FontSize',12);
set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:0.01:0.06],'YTick',[-0.7:0.1:0.5],'TickDir','in','box','off');
set(gcf,'color','white')
box on;


%figure 4: energy flow of the solar energy
figure;
G_abs_filter=(G2abs+G3abs+G4abs)/G;
G_abs_pv=Gpvabs/G;
G_el=effm;
G_rel=1-G_abs_filter-G_abs_pv-G_el;
Gflow=[G_abs_filter,G_abs_pv,G_el,G_rel];
labels={'Filter','PV-th','PV-el','Reflective Loss'};

p=pie(Gflow);

legend(labels,'Location','southoutside','Orientation','horizontal');
set(gcf,'color','white')

title('Solar Energy Flow','FontSize',15);

%figure 5: Temperature distributions

figure;
x0=0;
y0=800;
width=560*1.5;
height=420*1.5;
set(gcf,'position',[x0,y0,width,height]);


Plottem(1)=plot(Tr,Tsspvt(:,6)-273,'-o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Tpv-SS}');hold on;
Plottem(2)=plot(Tr,Tsspvt(:,4)-273,'--o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Tout-filter-SS}');hold on;
Plottem(3)=plot(Tr,Tsspvt(:,9)-273,'--p','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Tout-bottom-SS}');hold on;

Plottem(4)=plot(Tr1,Tcpvt(:,2)-273,'-o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Tpv-C}');hold on;
Plottem(5)=plot(Tr1,Tcpvt(:,5)-273,'--o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Tout-C}');hold on;



grid on;

axis([0 0.06 30 100]);
xlabel('{T}_{reduced} [-]','FontSize',16);
ylabel('{Temperature [C]}','FontSize',16);
legendtem=legend(Plottem,'Location','northeast');legend boxon;
set(legendtem,'FontSize',12);
set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:0.01:0.06],'YTick',[30:10:100],'TickDir','in','box','off');
set(gcf,'color','white')
box on;



%figure 6: critical values
figure;
x0=500;
y0=800;
width=560*1.5;
height=420*1.5;
set(gcf,'position',[x0,y0,width,height]);


Plotcrit(1)=plot(GG,Tr_el_crit,'-o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Electrical-critical Tr}');hold on;
Plotcrit(2)=plot(GG,Tr_th_crit,'--o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Thermal-critical Tr}');

grid on;

axis([0 1200 0 0.2]);
xlabel('{G} [W/m2]','FontSize',16);
ylabel('{T}_{reduced} [-]','FontSize',16);
legendcrit=legend(Plotcrit,'Location','northeast');legend boxon;
set(legendcrit,'FontSize',12);
set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:200:1200],'YTick',[0:0.02:0.2],'TickDir','in','box','off');
set(gcf,'color','white')
box on;


%figure 7: critical values
figure;
x0=1000;
y0=800;
width=560*1.5;
height=420*1.5;
set(gcf,'position',[x0,y0,width,height]);

for k=1:NG(2)
   T_el_crit(k)=GG(k)*Tr_el_crit(k)+Ta-273;
   T_th_crit(k)=GG(k)*Tr_th_crit(k)+Ta-273;
end

Plotcrit(1)=plot(GG,T_el_crit,'-o','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Electrical-critical Tm}');hold on;
Plotcrit(2)=plot(GG,T_th_crit,'--o','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{Thermal-critical Tm}');

grid on;

axis([0 1200 30 100]);
xlabel('{G} [W/m2]','FontSize',16);
ylabel('{Tm} [C]','FontSize',16);
legendcrit=legend(Plotcrit,'Location','northeast');legend boxon;
set(legendcrit,'FontSize',12);
set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:200:1200],'YTick',[30:10:100],'TickDir','in','box','off');
set(gcf,'color','white')
box on;



%figure: filters

% figure;
% x0=0;
% y0=0;
% width=560*1.5;
% height=420*1.5;
% set(gcf,'position',[x0,y0,width,height]);
% 
% x_G=xlsread('data', 'ambient conditions', 'B2:B2003');
% Tr_water=xlsread('data', 'optical properties', 'C3:C2004')/100;
% Tr_lowiron=xlsread('data', 'optical properties', 'X3:X2004')/100;
% Tr_ideal_Si=xlsread('data', 'optical properties', 'AJ3:AJ2004')/100;
% Tr_ideal_CdTe=xlsread('data', 'optical properties', 'AD3:AD2004')/100;
% Tr_shortpass=xlsread('data', 'optical properties', 'R3:R2004')/100;
% Tr_heatabsorb=xlsread('data', 'optical properties', 'M3:M2004')/100;
% 
% 
% 
% Plotfilter(1)=plot(x_G,Tr_ideal_Si,'-','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{ideal filter for Si}');hold on;
% Plotfilter(2)=plot(x_G,Tr_water,'--','Color',[0 0 0],'linewidth',2,'DisplayName','{Water-10mm}');
% 
% % Plotfilter(1)=plot(x_G,Tr_ideal_CdTe,'-','Color',[0.8500 0.3250 0.0980],'linewidth',2,'DisplayName','{ideal filter for CdTe}');hold on;
% % Plotfilter(2)=plot(x_G,Tr_water,'--','Color',[0 0 0],'linewidth',2,'DisplayName','{Water-10mm}');hold on;
% % Plotfilter(3)=plot(x_G,Tr_shortpass,'--','Color',[0 0.4470 0.7410],'linewidth',2,'DisplayName','{Short pass glass}');hold on;
% % Plotfilter(4)=plot(x_G,Tr_heatabsorb,'--','Color',[0 1 1],'linewidth',2,'DisplayName','{Heat-absorbing glass}');
% 
% 
% axis([0 2500 0 1.1]);
% xlabel('{Wavelength} [nm]','FontSize',16);
% ylabel('{Transmissivity [-]}','FontSize',14);
% legendfilter=legend(Plotfilter,'Location','northeast');legend boxon;
% set(legendfilter,'FontSize',18);
% set(gca,'Layer','Top','XColor','k','YColor','k','linewidth',0.5,'FontSize',16,'XTick',[0:500:2500],'YTick',[0:0.1:1.1],'TickDir','in','box','off');
% set(gcf,'color','white')
% box on;


