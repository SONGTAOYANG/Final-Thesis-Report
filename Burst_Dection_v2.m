% 1. Import data
data = readtable('/MATLAB Drive/energy_frequency_data_channel1_test1_SRUKF.xlsx');
% Extract required columns
time = data.Time; % Assuming 'Time' is the column name in the table
time = time(end-100:end);
total_energy = data.TotalEnergy;
total_energy = total_energy(end-100:end);
% Ensure time is in datetime format
if ~isdatetime(time)
    time = datetime(time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); % Adjust based on actual format
end

% Define RI calculation related functions
fir_filter = [-1 -1 -1 1 1 1];

% Modify RI calculation related functions
function g = weighting_function(n)
    weights = exp(-0.8 * (n-1));
    g = weights / sum(weights);
end

function s = scaling_function(x)
    s = 100 / (1 + exp(-0.8 * x));
end

function RI = calculate_RI_new(log_energies)
    % Calculate short-term and long-term energy change rates
    short_term_change = log_energies(end) - log_energies(end-1);
    long_term_change = log_energies(end) - mean(log_energies(1:end-1));
    % Calculate weighted average change rate
    weighted_change = 0.7 * short_term_change + 0.3 * long_term_change;
    % Map to RI value
    RI = scaling_function(weighted_change);
end

% Calculate logarithmic energy
log_total_energy = log10(total_energy);

% Initialize RI value array
ri_values = zeros(size(log_total_energy));

% Calculate RI values
window_size = 4; % Reduce window size to improve response speed
for i = window_size:length(log_total_energy)
    window_energies = log_total_energy(i-window_size+1:i);
    ri_values(i) = calculate_RI_new(window_energies);
end

% Adjust array lengths to match
time = time(window_size:end);
total_energy = total_energy(window_size:end);
log_total_energy = log_total_energy(window_size:end);
ri_values = ri_values(window_size:end);

% Convert total energy to logarithmic scale
log_total_energy = log10(total_energy);

% Plotting
figure;
subplot(2,1,1);
semilogy(time, total_energy); % Use semilogy for logarithmic scale plotting
title('Total Energy Over Time (Log Scale)');
xlabel('Time');
ylabel('Total Energy (Log Scale)');
grid on;

subplot(2,1,2);
plot(time, ri_values);
title('Responsiveness Index (RI) Over Time');
xlabel('Time');
ylabel('RI Value');
ylim([0 100]);
grid on;

% Export data to Excel, including logarithmic energy
data_to_export = table(time, total_energy, log_total_energy, ri_values, ...
    'VariableNames', {'Time', 'TotalEnergy', 'LogTotalEnergy', 'RI_Value'});
writetable(data_to_export, 'ri_data_output.xlsx');
disp('Data has been exported to ri_data_output.xlsx');