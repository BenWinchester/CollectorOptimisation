%% Spectral splitting PVT steady-state code (3/6/2019)

%The structure of the S-PVT
%-------------GLASS-----------------   T1
%             AIR GAP                  
%=========FILTER TOP GLASS==========   T2
%          SELCTIVE LIQUID             T3:inlet  T4:outlet
%========FILTER BOTTOM GLASS========   T5
%             AIR GAP                  
%---------------PV------------------   T6
%-------------Absorber--------------   T7
%             COOLANT                  T8:inlet  T9:outlet
%-----------INSULATION--------------

function [eff_th_fluid, eff_th_cool, eff_th_total, eff_el, T_r, T_sspvt, Energy, P_el, P_th] = sspvt_performance(Ta, Tcin, Tflin, GG, Vwind, Emgc, Emgfl1, Emgfl2, Abpv, Empv, Wpv, Betapv, Etapvref, Keva, Weva, Kad, Wad, Kinsul, Winsul, titanl, Wgap1, Wgap2, Wgap3, Wgap4, mc, mfl);
% SSPVT_PERFORMANCE  Calculate the performance of an SSPVT collector
%
% [eff_th_fluid, eff_th_cool, eff_th_total, eff_el, T_r, T_sspvt, Energy,
% P_el, P_th] = sspvt_performance()
%   Uses default parameters
%
% [eff_th_fluid, eff_th_cool, eff_th_total, eff_el, T_r, T_sspvt, Energy,
% P_el, P_th] = sspvt_performance(Ta, Tcin, Tflin, GG, Vwind)
%   Uses default design parameters and only varies the
%   environmental/operational parameters.
%
% [eff_th_fluid, eff_th_cool, eff_th_total, eff_el, T_r, T_sspvt, Energy,
% P_el, P_th] = sspvt_performance(Ta, Tcin, Tflin, GG, Vwind, Emgc, Emgfl1,
% Emgfl2, Abpv, Empv, Wpv, Betapv, Etapvref, Keva, Weva, Kad, Wad, Kinsul,
% Winsul, titanl, Wgap1, Wgap2, Wgap3, Wgap4)
%   Uses default values for the mass flow rates through the collector
%   layers as being based on the area of the collector
%
% [eff_th_fluid, eff_th_cool, eff_th_total, eff_el, T_r, T_sspvt, Energy,
% P_el, P_th] = sspvt_performance(Ta, Tcin, Tflin, GG, Vwind, Emgc, Emgfl1,
% Emgfl2, Abpv, Empv, Wpv, Betapv, Etapvref, Keva, Weva, Kad, Wad, Kinsul,
% Winsul, titanl, Wgap1, Wgap2, Wgap3, Wgap4, mc, mfl)
%   Uses all of the parameters specified
%

    %Geometry parameters
    Lc=1.2; % length - Aperture length, m                           
    Wc=0.6; % Aperture width, m
    Nc=1; % number of collectors
    Ac=Lc*Wc*Nc; %Lc*Wc; % aperture area, m2

    % Default values for the parameters
    if nargin < 26
        Emgc=0.9; % emissivity of the cover glass
        Emgfl1=0.04; % emissivity of the top glass of the filter
        Emgfl2=0.04; % emissivity of the bottom glass of the filter
        Empv=0.9; % emissivity of pv
        Abpv=0.93; % absorptivity of pv
        Betapv=false;
        Etapvref=false;
        Weva=0.0005;
        Keva=0.35;
        Wad=0.0005;
        Kad=0.85;
        Kinsul=0.005;%thermal conductivity of the insulation layer

        titanl=35*pi/180; % title angle of collector, rad%
        Wgap1=0.01;%The gap thickness between top glass and the filter
        Wgap2=0.01;%The gap thickness of the filter channel
        Wgap3=0.01;%The gap thickness between the filter and the PV
        Wgap4=0.01;%The gap thickness of the coolant channel 
        Winsul=0.04;%The thickness of the insulation layer
        if nargin < 5
            GG=[200 400 600 800 1000];
            Vwind=[1, 1, 1, 1, 1];
            Ta=[300, 300, 300, 300, 300]; %Ambient temperature K
            Tcin=[300, 300, 300, 300, 300];%bottom coolant inlet temperature 
            Tflin=[300,310,320,330,345];%filter coolant inlet temperature
        end
        mfl=repmat(0.01*Ac, 1, length(Ta));%mass flow rate of the filter fluid
        mc=repmat(0.01*Ac, 1, length(Ta));%mass flow rate of the bottom coolant
    end

    Tsky=0.0552.*Ta.^1.5; % sky temp
 
    
    %Optical proerties
    %select glss for each
    %layer:low_iron_glass,heat_absorb_glass,pure_water_10mm,anti_reflective_thin_glass,short_pass_glass,ideal_filter_glass_CdTe,ideal_filter_glass_Si
    
    %cover glass
    [Trgc,Relgc,Abgc]=select_filter('low_iron_glass');
    
    %top glass of the filter
    [Trgfl1,Relfl1,Abgfl1]=select_filter('anti_reflective_thin_glass');
    %Trgfl1=Trgfl1*(0.92/0.975);%change the transmissivity of the glass
    
    %bottom glass of the filter
    [Trgfl2,Relfl2,Abgfl2]=select_filter('anti_reflective_thin_glass');
    %Trgfl2=Trgfl2*(0.92/0.975);%change the transmissivity of the glass
    
    %working fluid in the filter
    [Trwt,Relwt,Abwt]=select_filter('pure_water_10mm');
    
   
    
    %physical properties
    cpfl=4180;%heat capacity of the fluid in the filter
    cpc=4180;%heat capacity of the fluid in the bottom coolant channel
    
    
    %Ambient conditions
    %G=1000;%Solar intensity W/m2
    AM=xlsread('data', 'ambient conditions', 'C2:C2003'); %the distribution of AM1.5
    GAM=1000.37;%The intensity of the standard AM1.5
    
    %****
    %====================CALCULATION=============================
    for j=1:length(GG)
    
        G=GG(j);    
        %Spectrum spliting on each layer
        G0=G/GAM*AM;%original incident full-spectrum solar
        
        [G1pass,G1rel,G1abs]=Gfilter(G0,Trgc,Relgc,Abgc);%layer1:cover glass,Specrum after the cover glass
        [G2pass,G2rel,G2abs]=Gfilter(G1pass,Trgfl1,Relfl1,Abgfl1);%layer2:top glass of filter,Spctrum after the top glass of the filter
        [G3pass,G3rel,G3abs]=Gfilter(G2pass,Trwt,Relwt,Abwt);%layer3:water,Spctrum after the water
        [G4pass,G4rel,G4abs]=Gfilter(G3pass,Trgfl2,Relfl2,Abgfl2);%layer4:bottom glass of filter,Spctrum after the bottom glass of the filter, G4pass is the spectrum arrived the pv
        
        %each layer will absorb the reflective spectrum
        [G4pass1,G4rel1,G4abs1]=Gfilter(G4pass*(1-Abpv),Trgfl2,Relfl2,Abgfl2);%bottom glass of filter
        G4abs=G4abs+G4abs1;
        
        [G3pass1,G3rel1,G3abs1]=Gfilter(G4rel+G4pass1,Trwt,Relwt,Abwt);%water
        G3abs=G3abs+G3abs1;
        
        [G2pass1,G2rel1,G2abs1]=Gfilter(G3rel+G3pass1,Trgfl1,Relfl1,Abgfl1);%top glass of filter
        G2abs=G2abs+G2abs1;
        
        [G1pass1,G1rel1,G1abs1]=Gfilter(G2rel+G2pass1,Trgc,Relgc,Abgc);
        G1abs=G1abs+G1abs1;
        
        
        % the performance of PV cell
        Solarcell=solarcell(Ac,G4pass*Abpv,G,'CdTe');

        % override the values if the user has specified performance values
        if ~(Betapv == false)
            Solarcell(6) = Betapv;
        end
        
        if ~(Etapvref == false)
            Solarcell(1) = Etapvref;
        end
    
        
        %simulation initials and settings
        Asol=zeros(9,9); % 9 Equations' left-hand coefficients
        Bsol=zeros(9,1); % 9 equations' right-hand coefficients
        T=zeros(1,9)+300;% T1...T9:Please see the structure of the S-PVT
        X=zeros(9,1)+300;% solution of each iteration
        Ttemp=9999;
        N=1;
        error=1e-4;
        
        while (max(abs(X.'-Ttemp))>error && N<100); 
            Ttemp=T;

            % heat transfer coefficients calculations
            [Kair1,vair1,Bair1,Prair1]=AirProperties(0.5*T(1)+0.5*T(2)); %air gap between glaze and filter
            [Kair2,vair2,Bair2,Prair2]=AirProperties(0.5*T(5)+0.5*T(6)); % air gap between filter and PV
            
            hr_g2sky=h_radiation(Emgc,0,T(1),Tsky(j)); % radiation from glaze to sky
            hwind=2.8+3*Vwind(j); %8.3+2.2*vwind(i-1); % 4.5+2.9*vwind, convection heat transfer due to wind
            
            hr_fl2g=h_radiation(Emgfl1,Emgc,T(2),T(1)); % radiation between cover glass and filter
            hc_fl2g=1/(2/h_enclosure(T(1),T(2),Wgap1,titanl)+Wgap1/Kair1); % convection and conduction between cover and glass through air gap 
            
            hr_fl2pv=h_radiation(Empv,Emgfl2,T(6),T(5)); % radiation between filter and PV
            % ATTENTION!!
            % I believe that I have found an error in Gan's code, whereby
            % the conduction and convection _between_ the filter and the
            % PV, through the gap, were utilising the width of the filter
            % channel, rather than the width of the gap, which was ending
            % up as an unused variable at the end of the code.
            % Original code:
            % hc_fl2pv=1/(2/h_enclosure(T(5),T(6),Wgap2,titanl)+Wgap2/Kair2); % convection and conduction between filter and pv through air gap 
            hc_fl2pv=1/(2/h_enclosure(T(5),T(6),Wgap2,titanl)+Wgap3/Kair2); % convection and conduction between filter and pv through air gap 
            
            hc_pv2ab=1/(Weva/Keva+Wad/Kad+0.0001/0.2);%the thermal resistance of the EVA, adhesive and ted. 
            
            
            hc_filter=h_watergap(T(3),T(4),mfl(j),Wgap2,Wc,Lc);%the convection coefficienct in the filter channel
            
            hc_cool=h_watergap(T(8),T(9),mc(j),Wgap4,Wc,Lc);%the convection coefficienct in the coolant channel
            
            h_cool2amb=1/(1/hc_cool+Winsul/Kinsul+1/hwind);%the heat transfer coefficient from the coolant fluid to the ambient
            
            
            %Equations:
            
            %Equation 1: cover glass
            %Ac*hr_g2sky*(Tsky-Tg)+Ac*hwind*(Ta-Tg)+Ac*(hr_fl2g+hc_fl2g)*(Tflg1-Tg)+Ac*Abgc*G=0
            
            Asol(1,1)=-Ac*hr_g2sky-Ac*hwind-Ac*(hr_fl2g+hc_fl2g);
            Asol(1,2)=Ac*(hr_fl2g+hc_fl2g);
            
            Bsol(1)=-Ac*hr_g2sky*Tsky(j)-Ac*hwind*Ta(j)-Ac*G1abs;
            
            
            %Equation 2: top glass of the filter
            %Ac*(hr_fl2g+hc_fl2g)*(Tg-Tflg1)+Ac*hc_filter*(0.5*Tflin+0.5Tflout-Tflg1)+Ac*G2abs=0
            
            Asol(2,1)=Ac*(hr_fl2g+hc_fl2g);
            Asol(2,2)=-Ac*(hr_fl2g+hc_fl2g)-Ac*hc_filter;
            Asol(2,3)=Ac*hc_filter*0.5;
            Asol(2,4)=Ac*hc_filter*0.5;
            
            Bsol(2)=-Ac*G2abs;
            
            %Equation 3: water layer
            %Ac*hc_filter*(Tflg1-0.5*Tflin-0.5Tflout)+Ac*hc_filter*(Tflg2-0.5*Tflin-0.5Tflout)+Ac*G3abs-mfl*cpfl*(Tflout-Tflin)=0
            
            Asol(3,2)=Ac*hc_filter;
            Asol(3,3)=-Ac*hc_filter*0.5-Ac*hc_filter*0.5+mfl(j)*cpfl;
            Asol(3,4)=-Ac*hc_filter*0.5-Ac*hc_filter*0.5-mfl(j)*cpfl;
            Asol(3,5)=Ac*hc_filter;
            
            Bsol(3)=-Ac*G3abs;
            
            %Equation 4: bottom glass of the filter
            %Ac*hc_filter*(0.5*Tflin+0.5Tflout-Tflg2)+Ac*(hr_fl2pv+hc_fl2pv)*(Tpv-Tflg2)+Ac*G4abs=0
            
            Asol(4,3)=Ac*hc_filter*0.5;
            Asol(4,4)=Ac*hc_filter*0.5;
            Asol(4,5)=-Ac*hc_filter-Ac*(hr_fl2pv+hc_fl2pv);
            Asol(4,6)=Ac*(hr_fl2pv+hc_fl2pv);
            
            Bsol(4)=-Ac*G4abs;
            
            %Equation 5: PV
            %Ac*(hr_fl2pv+hc_fl2pv)*(Tflg2-Tpv)+Ac*hc_pv2ab*(Tabs-Tpv)+Ac*Gpvabs=0
            effm=Solarcell(1)*(1-Solarcell(6)*(T(6)-273.15-25));%module efficiency
            Gpvabs=Solarcell(5)-effm*G;%waste heat in PV
            
            
            Asol(5,5)=Ac*(hr_fl2pv+hc_fl2pv);
            Asol(5,6)=-Ac*(hr_fl2pv+hc_fl2pv)-Ac*hc_pv2ab;
            Asol(5,7)=Ac*hc_pv2ab;
            
            Bsol(5)=-Ac*Gpvabs;
            
            %Equation 6:absorber
            %Ac*hc_pv2ab*(Tpv-Tabs)+Ac*hc_cool*(0.5*Tcout+0.5*Tcin-Tabs)=0
            
            Asol(6,6)=Ac*hc_pv2ab;
            Asol(6,7)=-Ac*hc_pv2ab-Ac*hc_cool;
            Asol(6,8)=Ac*hc_cool*0.5;
            Asol(6,9)=Ac*hc_cool*0.5;
            
            Bsol(6)=0;
            
            %Equation 7:cooling channel
            %Ac*hc_cool*(Tabs-0.5*Tcout-0.5*Tcin)+Ac*h_cool2amb*(Ta-0.5*Tcout-0.5*Tcin)-mc*cpc*(Tcout-Tcin)=0
            
            Asol(7,7)=Ac*hc_cool;
            Asol(7,8)=-Ac*hc_cool*0.5-Ac*h_cool2amb*0.5+mc(j)*cpc;
            Asol(7,9)=-Ac*hc_cool*0.5-Ac*h_cool2amb*0.5-mc(j)*cpc;
            
            Bsol(7)=-Ac*h_cool2amb*Ta(j);
            
            %Equation 8 and 9: Tci=Tci,Tflin=Tflin or Tflin=Tcout;
            
            Asol(8,3)=1;
            Bsol(8)=Tflin(j);
            
            Asol(9,8)=1;
            Bsol(9)=Tcin(j);
            
            %****LINEAR SOLVER****
            X=linsolve(Asol(1:9,1:9),Bsol(1:9,1:1));
            T=X.';
            
            N=N+1;%count the iteration times
        
        end
    
        %efficiencies
        T_r(j)=(0.5*T(3)+0.5*T(4)-Ta(j))/G;
        eff_th_fluid(j)=mfl(j)*cpfl*(T(4)-T(3))/G/Ac;
        eff_th_cool(j)=mc(j)*cpc*(T(9)-T(8))/G/Ac;
        eff_th_total(j)=eff_th_fluid(j)+eff_th_cool(j);
        eff_el(j)=effm;
        
        %temperatures
        T_sspvt(j,1)=T(1);
        T_sspvt(j,2)=T(2);
        T_sspvt(j,3)=T(3);
        T_sspvt(j,4)=T(4);
        T_sspvt(j,5)=T(5);
        T_sspvt(j,6)=T(6);
        T_sspvt(j,7)=T(7);
        T_sspvt(j,8)=T(8);
        T_sspvt(j,9)=T(9);
        
        %energy balance
        Energy(j,1)=hr_g2sky*(T(1)-Tsky(j))/G;
        Energy(j,2)=hwind*(T(1)-Ta(j))/G;
        Energy(j,3)=(G2abs+G3abs+G4abs)/G;
        Energy(j,4)=hr_fl2g*(T(2)-T(1))/G;
        Energy(j,5)=hc_fl2g*(T(2)-T(1))/G;
        Energy(j,6)=hr_fl2pv*(T(5)-T(6))/G;
        Energy(j,7)=hc_fl2pv*(T(5)-T(6))/G;
        Energy(j,8)=mfl(j)*cpfl*(T(4)-T(3))/G/Ac;
        Energy(j,9)=hc_cool*(T(7)-0.5*T(8)-0.5*T(9))/G;
        Energy(j,10)=h_cool2amb*(0.5*T(8)+0.5*T(9)-Ta(j))/G;
        Energy(j,11)=mc(j)*cpfl*(T(9)-T(8))/G/Ac;
        
        % Power output
        P_el(j)=effm*G*Ac;
        P_th(j)=mc(j)*cpfl*(T(9)-T(8)) + mfl(j)*cpfl*(T(4)-T(3));
   
    end
end