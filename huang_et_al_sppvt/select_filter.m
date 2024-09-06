function [Tr,Rel,Abs]=select_filter(glass);

    if strcmp(glass,'low_iron_glass')==1
      Tr=table2array(readtable('data', sheet='optical properties',range= 'H3:H2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='I3:I2004'))/100; 
      Abs=table2array(readtable('data', sheet='optical properties', range='J3:J2004'))/100; 
    end

    if strcmp(glass,'heat_absorb_glass')==1
      Tr=table2array(readtable('data', sheet='optical properties', range='M3:M2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='N3:N2004'))/100; 
      Abs=table2array(readtable('data', sheet='optical properties', range='O3:O2004'))/100; 
    end

    if strcmp(glass,'pure_water_10mm')==1
      Tr=table2array(readtable('data', sheet='optical properties', range='C3:C2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='D3:D2004'))/100;
      Abs=table2array(readtable('data', sheet='optical properties', range='E3:E2004'))/100; 
    end
    
     if strcmp(glass,'anti_reflective_thin_glass')==1
      Tr=table2array(readtable('data', sheet='optical properties', range='X3:X2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='Y3:Y2004'))/100;
      Abs=table2array(readtable('data', sheet='optical properties', range='Z3:Z2004'))/100; 
     end
  
     if strcmp(glass,'short_pass_glass')==1
      Tr=table2array(readtable('data', sheet='optical properties', range='R3:R2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='S3:S2004'))/100;
      Abs=table2array(readtable('data', sheet='optical properties', range='T3:T2004'))/100; 
     end
     
      if strcmp(glass,'ideal_filter_glass_CdTe')==1
      Tr=table2array(readtable('data', sheet='optical properties', range='AD3:AD2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='AE3:AE2004'))/100;
      Abs=table2array(readtable('data', sheet='optical properties', range='AF3:AF2004'))/100; 
      end
     
     if strcmp(glass,'ideal_filter_glass_Si')==1
      Tr=table2array(readtable('data', sheet='optical properties', range='AJ3:AJ2004'))/100; 
      Rel=table2array(readtable('data', sheet='optical properties', range='AK3:AK2004'))/100;
      Abs=table2array(readtable('data', sheet='optical properties', range='AL3:AL2004'))/100; 
     end
     

end