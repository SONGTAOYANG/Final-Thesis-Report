% 1. Import data
data = readtable('energy_frequency_data_channel1_test1_SRUKF.xlsx');
% Extract required columns and remove the first element
time = data.Time(2:end);
mean_dominant_freq = data.MeanDominantFreq_Method2(2:end);
% Ensure time is in datetime format
if ~isdatetime(time)
    time = datetime(time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); % Adjust based on actual format
end

% 2. Wavelet transform smoothing
% Define wavelet transform smoothing function
function smoothed = wavelet_smooth(signal, wavelet_name, level)
    [c,l] = wavedec(signal, level, wavelet_name);
    threshold = median(abs(c))/0.6745 * sqrt(2*log(length(signal)));
    c_t = wthresh(c, 's', threshold);
    smoothed = waverec(c_t, l, wavelet_name);
end

% Apply wavelet transform smoothing to the signal
wavelet_name = 'db4'; % Use Daubechies 4 wavelet
level = 5; % Decomposition level
smoothed_mean_dominant_freq = wavelet_smooth(mean_dominant_freq, wavelet_name, level);

% 3. Plot the smoothed time series
figure('Position', [100, 100, 1200, 500]);
plot(time, smoothed_mean_dominant_freq, 'Color', '#367DB0', 'LineWidth', 1)
ylabel('Frequency (Hz)', 'FontSize', 10)
xlabel('Time', 'FontSize', 10)
title('Smoothed Time Series of Mean Dominant Frequency', 'FontSize', 12)
grid on
ax = gca;
ax.FontSize = 9; % Set axis tick font size

% Modify x-axis display
ax.XAxis.TickLabelFormat = 'HH:mm';
ax.XLim = [time(1), time(end)];

% Adjust figure layout
set(gca, 'Position', [0.1, 0.1, 0.8, 0.8]) % Adjust plotting area