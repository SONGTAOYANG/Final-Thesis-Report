% 1. Import data
data = readtable('/MATLAB Drive/energy_frequency_data_channel2_test1_SRUKF.xlsx');
% Extract required columns
time = data.Time; 
%time = time(100:500);
total_energy = data.TotalEnergy;
%total_energy = total_energy(100:500);
% Ensure time is in datetime format
if ~isdatetime(time)
    time = datetime(time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); % Adjust based on actual format
end

% Define window size
window_size = 7;

% weighting function
function weights = new_weighting_function()
    % Define parameters for exponential decay function
    B = 0.3; % Increase decay rate to reduce weight sum of first 3 points
    n = 0:5; % Index array (6 points)
    % Calculate unnormalized weights
    unnormalized_weights = exp(-B * n);
    % Normalize weights
    weights = unnormalized_weights / sum(unnormalized_weights);
    % Check sum of weights for first 3 points
    front_sum = sum(weights(1:3));
    fprintf('Sum of weights for first 3 points: %.4f\n', front_sum);
end

% Modified scaling function
function s = scaling_function(x)
    if numel(x) > 1
        warning('Input to scaling_function is not a scalar. Using mean value.');
        x = mean(x);
    end
    s = 100 / (1 + exp(-0.8 * x));
end

% Modified RI calculation function
function RI = calculate_RI_new(log_energies)
    % Get weights
    weights = new_weighting_function();
    % Ensure correct length of log_energies
    if length(log_energies) ~= 7
        error('log_energies must have length 7');
    end
    % Calculate short-term energy change rate
    short_term_change = log_energies(end) - log_energies(end-1);
    % Calculate weighted long-term energy change rate
    long_term_changes = log_energies(end) - log_energies(1:end-1);
    weighted_long_term_change = sum(weights .* long_term_changes);
    % Calculate weighted average change rate
    weighted_change = 0.7 * short_term_change + 0.3 * weighted_long_term_change;
    % Map to RI value
    RI = scaling_function(weighted_change);
end

% Calculate logarithmic energy
log_total_energy = log10(total_energy);

% Initialize RI value array
ri_values = zeros(size(log_total_energy));

% Calculate RI values
for i = window_size:length(log_total_energy)
    window_energies = log_total_energy(i-window_size+1:i);
    ri_values(i) = calculate_RI_new(window_energies);
end

% Adjust array lengths to match
time = time(window_size:end);
total_energy = total_energy(window_size:end);
log_total_energy = log_total_energy(window_size:end);
ri_values = ri_values(window_size:end);

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
title('Burst Index (BI) Over Time');
xlabel('Time');
ylabel('BI Value');
ylim([0 100]);
grid on;

% Export data to Excel
data_to_export = table(time, total_energy, log_total_energy, ri_values, ...
    'VariableNames', {'Time', 'TotalEnergy', 'LogTotalEnergy', 'RI_Value'});
writetable(data_to_export, 'ri_data_output.xlsx');
disp('Data has been exported to ri_data_output.xlsx');
