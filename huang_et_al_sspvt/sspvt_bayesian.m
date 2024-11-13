% Parameters that are passed in:
%   panel_filename, which is the filename of the panel data file;
%   suffix, which is the suffix used for the temporary weather-data files.

%% Open the temporary file for the panel.
panel_data = jsondecode(fileread(fullfile("temp", panel_filename)));
disp("Panel data parsed.")

% Open the weather-data inputs
Ta = csvread(fullfile("temp", "temp_ambient_temperature_inputs_" + num2str(suffix) + ".csv"));
Tcin = csvread(fullfile("temp", "temp_coolant_temperature_inputs_" + num2str(suffix) + ".csv"));
Tflin = csvread(fullfile("temp", "temp_fluid_temperature_inputs_" + num2str(suffix) + ".csv"));
GG = csvread(fullfile("temp", "temp_irradiance_inputs_" + num2str(suffix) + ".csv"));
Vwind = csvread(fullfile("temp", "temp_wind_speed_inputs_" + num2str(suffix) + ".csv"));


Ta = Ta + 273.15;

% Pass all of this information to the SSPVT script and save the outputs
[eff_th_fluid, eff_th_cool, eff_th_total, eff_el, T_r, T_sspvt, Energy, P_el, P_th, P_fl, P_cool, P_in] = sspvt_performance(...
    panel_data.glass_emissivity,...
    panel_data.filter_glass_emissivity,...
    panel_data.filter_glass_emissivity,...
    panel_data.pv_absorptivity,...
    panel_data.pv_emissivity,...
    panel_data.pv_thickness,...
    panel_data.pv_thermal_coefficient,...
    panel_data.pv_solar_cell_material,...
    panel_data.eva_thermal_conductivity,...
    panel_data.eva_thickness,...
    panel_data.adhesive_thermal_conductivity,...
    panel_data.adhesive_thickness,...
    panel_data.insulation_thermal_conductivity,...
    panel_data.insulation_thickness,...
    panel_data.tilt_angle,...
    panel_data.top_glass_to_filter_gap,...
    panel_data.filter_width,...
    panel_data.filter_to_pv_gap,...
    panel_data.coolant_width,...
    Ta,...
    Tcin,...
    Tflin,...
    GG,...
    Vwind...
);
disp("SSPV-T calculation complete.")

% Save the output data
output_data = struct();
output_data.eff_th_fluid = eff_th_fluid;
ouptut_data.eff_th_cool = eff_th_cool;
output_data.eff_th_total = eff_th_total;
output_data.eff_el = eff_el;
output_data.T_r = T_r;
output_data.T_sspvt = T_sspvt;
output_data.Energy = Energy;
output_data.P_el = P_el;
output_data.P_th = P_th;
output_data.P_fl = P_fl;
output_data.P_cool = P_cool;
output_data.P_in = P_in;

fid = fopen(fullfile("sspvt_bayesian_output", "results_run_" + num2str(suffix) + "_" + panel_filename + ".json"), 'w', 'n', 'UTF-8');
encoded_data = jsonencode(output_data);
fprintf(fid,'%s',encoded_data);
fclose(fid);
