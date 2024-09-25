function [Tr,Rel,Abs]=select_filter(glass);

    if strcmp(glass,'low_iron_glass')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties','Range',  'H3:H2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'I3:I2004'))/100; 
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'J3:J2004'))/100; 
    end

    if strcmp(glass,'heat_absorb_glass')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'M3:M2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'N3:N2004'))/100; 
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'O3:O2004'))/100; 
    end

    if strcmp(glass,'pure_water_10mm')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'C3:C2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'D3:D2004'))/100;
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'E3:E2004'))/100; 
    end
    
     if strcmp(glass,'anti_reflective_thin_glass')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'X3:X2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'Y3:Y2004'))/100;
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'Z3:Z2004'))/100; 
     end
  
     if strcmp(glass,'short_pass_glass')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'R3:R2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'S3:S2004'))/100;
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'T3:T2004'))/100; 
     end
     
      if strcmp(glass,'ideal_filter_glass_CdTe')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'AD3:AD2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'AE3:AE2004'))/100;
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'AF3:AF2004'))/100; 
      end
     
     if strcmp(glass,'ideal_filter_glass_Si')==1
      Tr=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'AJ3:AJ2004'))/100; 
      Rel=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'AK3:AK2004'))/100;
      Abs=table2array(readtable('data.xlsx', 'Sheet', 'optical properties', 'Range', 'AL3:AL2004'))/100; 
     end
     

end