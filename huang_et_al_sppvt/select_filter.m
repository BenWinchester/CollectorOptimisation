function [Tr,Rel,Abs]=select_filter(glass);

    if strcmp(glass,'low_iron_glass')==1
      Tr=xlsread('data', 'optical properties', 'H3:H2004')/100; 
      Rel=xlsread('data', 'optical properties', 'I3:I2004')/100; 
      Abs=xlsread('data', 'optical properties', 'J3:J2004')/100; 
    end

    if strcmp(glass,'heat_absorb_glass')==1
      Tr=xlsread('data', 'optical properties', 'M3:M2004')/100; 
      Rel=xlsread('data', 'optical properties', 'N3:N2004')/100; 
      Abs=xlsread('data', 'optical properties', 'O3:O2004')/100; 
    end

    if strcmp(glass,'pure_water_10mm')==1
      Tr=xlsread('data', 'optical properties', 'C3:C2004')/100; 
      Rel=xlsread('data', 'optical properties', 'D3:D2004')/100;
      Abs=xlsread('data', 'optical properties', 'E3:E2004')/100; 
    end
    
     if strcmp(glass,'anti_reflective_thin_glass')==1
      Tr=xlsread('data', 'optical properties', 'X3:X2004')/100; 
      Rel=xlsread('data', 'optical properties', 'Y3:Y2004')/100;
      Abs=xlsread('data', 'optical properties', 'Z3:Z2004')/100; 
     end
  
     if strcmp(glass,'short_pass_glass')==1
      Tr=xlsread('data', 'optical properties', 'R3:R2004')/100; 
      Rel=xlsread('data', 'optical properties', 'S3:S2004')/100;
      Abs=xlsread('data', 'optical properties', 'T3:T2004')/100; 
     end
     
      if strcmp(glass,'ideal_filter_glass_CdTe')==1
      Tr=xlsread('data', 'optical properties', 'AD3:AD2004')/100; 
      Rel=xlsread('data', 'optical properties', 'AE3:AE2004')/100;
      Abs=xlsread('data', 'optical properties', 'AF3:AF2004')/100; 
      end
     
     if strcmp(glass,'ideal_filter_glass_Si')==1
      Tr=xlsread('data', 'optical properties', 'AJ3:AJ2004')/100; 
      Rel=xlsread('data', 'optical properties', 'AK3:AK2004')/100;
      Abs=xlsread('data', 'optical properties', 'AL3:AL2004')/100; 
     end
     

end