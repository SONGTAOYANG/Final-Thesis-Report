% File path
file_path = '....hdf5';
% Open the HDF5 file
file_info = h5info(file_path);
% Get information about the RawData/Samples dataset
samples_info = h5info(file_path, '/RawData/Samples');
% Read the entire Samples dataset
samples_data = h5read(file_path, '/RawData/Samples');
samples_data_transposed = samples_data';
% Assuming data is stored in columns, each column corresponding to a channel
channel_1_data = samples_data_transposed(:, 1); % Data from the first channel
channel_2_data = samples_data_transposed(:, 2); % Data from the second channel

% Rename signal data
signal = channel_2_data;
% Sampling frequency
Fs = 4800; % in Hz 

%% Design and apply notch filters for 50 Hz and its harmonics
notch_freqs     = 50:50:200; % Frequencies to notch out (50 Hz, 100 Hz, 150 Hz, 200Hz)
BW_notch_freqs  = 1.5;
FilterOrder     = 10;

for f_notch = notch_freqs
    % Design the notch filter
    d = designfilt('bandstopiir', 'FilterOrder', FilterOrder, 'HalfPowerFrequency1', ...
                   f_notch-BW_notch_freqs, 'HalfPowerFrequency2', f_notch+BW_notch_freqs, 'SampleRate', Fs);
    % Apply the filter
    signal = filter(d, signal);
end

%% Design a bandpass filter
FilterOrder = 10;

low_cutoff  = 35;    % Lower cutoff frequency (Hz)
high_cutoff = 150;  % Upper cutoff frequency (Hz)
d = designfilt('bandpassiir', 'FilterOrder', FilterOrder, ...
               'HalfPowerFrequency1', low_cutoff, 'HalfPowerFrequency2', high_cutoff, ...
               'SampleRate', Fs);

% Apply the filter using filtfilt for zero-phase distortion
signal = filtfilt(d, double(signal));

%% Square Root Unscented Kalman Filter with Adaptive Noise Estimation and Dual Observation

% SR-UKF parameters
n = 2; % State dimension (amplitude and amplitude difference)
m = 2; % Measurement dimension 
alpha = 1e-3;
ki = 0;
beta = 2;
lambda = alpha^2 * (n + ki) - n;

% Weight calculations
Wm = [lambda / (n + lambda), repmat(1 / (2*(n + lambda)), 1, 2*n)];
Wc = Wm;
Wc(1) = Wc(1) + (1 - alpha^2 + beta);

% Initial state estimate and square root of covariance
x = [signal(1); 0]; % Initial guess: first amplitude and zero difference
S = chol(eye(n) * 0.1)'; % Square root of initial covariance, scaled down

% Initial process and measurement noise covariances
Q = diag([0.01, 0.001]); % Process noise for amplitude and difference, reduced
R = diag([0.1, 0.1]); % Measurement noise for amplitude and difference

% Parameters for adaptive noise estimation
window_size = 50;
innovation_history = zeros(m, window_size);

% SR-UKF loop
filtered_signal = zeros(size(signal));
for k = 1:length(signal)
    % Generate sigma points
    X = [x, x + sqrt(n + lambda) * S, x - sqrt(n + lambda) * S];
    
    % Generate process noise
    process_noise = mvnrnd([0; 0], Q, 2*n+1)';
    
    % Time update (prediction)
    X_pred = zeros(size(X));
    for i = 1:2*n+1
        X_pred(:,i) = [X(1,i) + X(2,i); X(2,i)] + process_noise(:,i);
    end
    x_pred = sum(Wm .* X_pred, 2);
    
    % Calculate predicted square root of covariance
    Y = zeros(n, 2*n+1);
    for i = 1:2*n+1
        Y(:,i) = sqrt(abs(Wc(i))) * (X_pred(:,i) - x_pred);
    end
    
    % Ensure Q is positive definite
    Q = (Q + Q') / 2;
    [V, D] = eig(Q);
    D = max(D, 1e-6);  % Ensure all eigenvalues are positive
    Q = V * D * V';
    Q = Q + 1e-6 * eye(size(Q));
    
    % Use SVD instead of Cholesky decomposition for more stability
    [U, S, V] = svd(Q);
    SQ = U * sqrt(S);
    
    S_pred = qr([Y, SQ]', 0)';
    S_pred = S_pred(1:n, 1:n);  % Ensure S_pred is square
    
    % Measurement update
    Z = X_pred; % 
    z_pred = sum(Wm .* Z, 2);
    
    % Calculate innovation square root of covariance
    Y = zeros(m, 2*n+1);
    for i = 1:2*n+1
        Y(:,i) = sqrt(abs(Wc(i))) * (Z(:,i) - z_pred);
    end
    
    % Ensure R is positive definite
    R = (R + R') / 2;
    [V, D] = eig(R);
    D = max(D, 1e-6);  % Ensure all eigenvalues are positive
    R = V * D * V';
    R = R + 1e-6 * eye(size(R));
    
    [U, S, V] = svd(R);
    SR = U * sqrt(S);
    
    Sz = qr([Y, SR]', 0)';
    Sz = Sz(1:m, 1:m);  % Ensure Sz is square
    
    % Calculate cross-covariance
    Pxz = zeros(n, m);
    for i = 1:2*n+1
        Pxz = Pxz + Wc(i) * (X_pred(:,i) - x_pred) * (Z(:,i) - z_pred)';
    end
    
    % Kalman gain and state update
    K = (Pxz / Sz') / Sz;
    innovation = [signal(k); signal(k) - signal(max(k-1,1))] - z_pred;
    x = x_pred + K * innovation;
    
    % Update square root of covariance
    U = K * Sz;
    S = qr([S_pred'; U']', 0)';
    S = S(1:n, 1:n);  % Ensure S is square
    
    % Store filtered value
    filtered_signal(k) = x(1);
    
    % Update innovation history
    innovation_history = [innovation_history(:,2:end), innovation];
    
    % Adaptive noise estimation
    if k >= window_size
        [Q, R] = adaptive_noise_estimation(innovation_history, S*S', window_size);
    end
end

% Replace original signal with filtered signal
signal = filtered_signal;

function [Q, R] = adaptive_noise_estimation(innovation, P, window_size)
    % innovation: Measurement residual history
    % P: State covariance matrix
    % window_size: Size of the sliding window

    % Estimate R
    R = cov(innovation');

    % Estimate Q
    S = cov(innovation');
    H = eye(2); % We now observe two state variables
    Q = S - H*P*H';

    % Ensure Q and R are positive definite
    Q = (Q + Q') / 2;
    R = (R + R') / 2;
    [V, D] = eig(Q);
    D = max(D, 1e-6); % Ensure all eigenvalues are positive
    Q = V * D * V';
    [V, D] = eig(R);
    D = max(D, 1e-6); % Ensure all eigenvalues are positive
    R = V * D * V';

    % Add small positive definite terms to ensure numerical stability
    Q = Q + 1e-6 * eye(size(Q));
    R = R + 1e-6 * eye(size(R));
end


%% Downsample the signal
downsample_factor = 10; % Downsampling factor (adjust as needed)
signal_ds = downsample(signal, downsample_factor);
Fs_ds = Fs / downsample_factor; % Adjusted sampling frequency

%% Live plot of wavelet
dataLength = length(signal_ds);
sec_per_fr = 2;
frameLength = sec_per_fr * Fs_ds;
t = 0:1/Fs_ds:sec_per_fr-1/Fs_ds;
numFrames = floor(dataLength / frameLength);
signal_dsReshaped = reshape(signal_ds(1:numFrames*frameLength), frameLength, []);

% Calculate actual number of frames
actualFrames = size(signal_dsReshaped, 2);

strFrame = 1;
addFrame = min(1800, actualFrames - 1);  % 确保不会超过实际的帧数
scale_max = 5;

% Define frequency bands
freq_bands = 35:5:150;
num_bands = length(freq_bands) - 1;

% Initialize arrays to store results
time_axis = zeros(1, actualFrames);
energy_ratio = zeros(1, actualFrames);
total_energy = zeros(1, actualFrames);
high_energy_array = zeros(1, actualFrames);
low_energy_array = zeros(1, actualFrames);
mean_frequencies = zeros(1, actualFrames);
dominant_frequencies = zeros(1, actualFrames);
mean_dominant_freq = zeros(1, actualFrames);
band_proportions = zeros(num_bands, actualFrames);
high_energy_ratio = zeros(1, actualFrames);
low_energy_ratio = zeros(1, actualFrames);

% Create a directory to store plots if it doesn't exist
if ~exist('plot4', 'dir')
    mkdir('plot4');
end

% Create time axis
start_time = datetime('8:09:00', 'InputFormat', 'H:mm:ss');
time_axis = start_time + seconds((0:addFrame) * sec_per_fr);
time_axis_num = datenum(time_axis);  % Convert datetime to numeric

for i = strFrame:min(strFrame+addFrame, actualFrames)
    % Wavelet transform
    [wt, f] = cwt(signal_dsReshaped(:, i), Fs_ds);
    
    % Calculate power spectrum
    power_spectrum = abs(wt).^2;
    
    % Calculate mean frequency
    total_power = sum(power_spectrum, 1);
    weighted_freq = sum(f .* power_spectrum, 1) ./ total_power;
    mean_frequencies(i) = mean(weighted_freq);
    
    % Method 1: Find dominant frequency
    [~, max_idx] = max(sum(power_spectrum, 2));
    dominant_frequencies(i) = f(max_idx);
    
    % Method 2: Calculate second type of dominant frequency
    [~, idx] = max(abs(wt), [], 1);
    dominant_freq = f(idx);
    mean_dominant_freq(i) = mean(dominant_freq);
    
    % Calculate power in each frequency band
    band_powers = zeros(1, num_bands);
    for j = 1:num_bands
        band_mask = (f >= freq_bands(j)) & (f < freq_bands(j+1));
        band_powers(j) = sum(sum(power_spectrum(band_mask, :)));
    end
    
    % Calculate proportions
    band_proportions(:, i) = band_powers / sum(band_powers);
    
    % Calculate total energy
    total_energy(i) = sum(power_spectrum, 'all');
    
    % Calculate high and low frequency energies
    high_freq_idx = (f >= 95) & (f <= 150);
    low_freq_idx = (f >= 35) & (f <= 65);
    high_energy = sum(power_spectrum(high_freq_idx, :), 'all');
    low_energy = sum(power_spectrum(low_freq_idx, :), 'all');
    
    % Store high_energy and low_energy
    high_energy_array(i) = high_energy;
    low_energy_array(i) = low_energy;
    
    % Calculate energy ratio
    energy_ratio(i) = high_energy / low_energy;
    
    % Calculate the proportion of high-frequency and low-frequency energy to total energy
    high_energy_ratio(i) = high_energy / total_energy(i);
    low_energy_ratio(i) = low_energy / total_energy(i);
end

% Prepare data for Excel export
data_to_export = table(time_axis', ...
                       energy_ratio', ...
                       total_energy', ...
                       high_energy_array', ...
                       low_energy_array', ...
                       high_energy_ratio', ...
                       low_energy_ratio', ...
                       mean_frequencies', ...
                       dominant_frequencies', ...
                       mean_dominant_freq', ...
    'VariableNames', {'Time', 'EnergyRatio', 'TotalEnergy', 'HighEnergy', 'LowEnergy', ...
                      'HighEnergyRatio', 'LowEnergyRatio', ...
                      'MeanFrequency', 'DominantFreq_Method1', 'MeanDominantFreq_Method2'});

% Add band proportions to the table
for j = 1:num_bands
    data_to_export.([sprintf('Band_%d_%d', freq_bands(j), freq_bands(j+1))]) = band_proportions(j, :)';
end

% Export data to Excel
writetable(data_to_export, 'energy_frequency_data_channel2_test1_SRUKF.xlsx');
disp('Data has been exported to energy_frequency_data_channel2_test1_SRUKF.xlsx');

% Calculate and display total time
timePast = addFrame * sec_per_fr / 60;   
X = ['Total time past is: ', num2str(timePast), ' minutes.'];
disp(X);
