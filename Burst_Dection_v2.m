% 1. 导入数据
data = readtable('/MATLAB Drive/energy_frequency_data_channel1_test1_SRUKF.xlsx');
% 提取所需列
time = data.Time; % 假设 'Time' 是表格中的列名
time = time(end-100:end);
total_energy = data.TotalEnergy;
total_energy = total_energy(end-100:end);
% 确保时间是 datetime 类型
if ~isdatetime(time)
    time = datetime(time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); % 根据实际格式调整
end

% 定义RI计算相关函数
fir_filter = [-1 -1 -1 1 1 1];

% 修改RI计算相关函数
function g = weighting_function(n)
    weights = exp(-0.8 * (n-1));
    g = weights / sum(weights);
end

%function s = scaling_function(x)
%    % 使用tanh函数进行缩放，并调整参数以更好地利用0-100范围
%    s = 50 * (tanh(x) + 1);
%end

function s = scaling_function(x)
 s = 100 / (1 + exp(-0.8 * x));
 
end
function RI = calculate_RI_new(log_energies)
    % 计算短期和长期能量变化率
    short_term_change = log_energies(end) - log_energies(end-1);
    long_term_change = log_energies(end) - mean(log_energies(1:end-1));
    
    % 计算加权平均变化率
    weighted_change = 0.7 * short_term_change + 0.3 * long_term_change;
    
    % 映射到RI值
    RI = scaling_function(weighted_change);
end

% 计算对数能量
log_total_energy = log10(total_energy);

% 初始化RI值数组
ri_values = zeros(size(log_total_energy));

% 计算RI值
window_size = 4;  % 减少窗口大小以提高响应速度
for i = window_size:length(log_total_energy)
    window_energies = log_total_energy(i-window_size+1:i);
    ri_values(i) = calculate_RI_new(window_energies);
end

% 调整数组长度以匹配
time = time(window_size:end);
total_energy = total_energy(window_size:end);
log_total_energy = log_total_energy(window_size:end);
ri_values = ri_values(window_size:end);

% 将总能量转换为对数尺度
log_total_energy = log10(total_energy);

% 绘图
figure;
subplot(2,1,1);
semilogy(time, total_energy); % 使用 semilogy 进行对数坐标绘图
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

% 导出数据到Excel，包括对数能量
data_to_export = table(time, total_energy, log_total_energy, ri_values, ...
    'VariableNames', {'Time', 'TotalEnergy', 'LogTotalEnergy', 'RI_Value'});
writetable(data_to_export, 'ri_data_output.xlsx');
disp('Data has been exported to ri_data_output.xlsx');