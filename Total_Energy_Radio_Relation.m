% Import data
data = readtable('energy_frequency_data_night2.xlsx');

% Extract required columns
time = data.Time;
total_energy = data.TotalEnergy;
energy_ratio = data.EnergyRatio;

% UKF parameters
n = 2; % State dimension (we're tracking total_energy and energy_ratio)
m = 2; % Measurement dimension
Q = diag([0.1, 0.1]); % Process noise covariance
R = diag([0.1, 0.1]); % Measurement noise covariance
P = eye(n); % Initial error covariance
x = [total_energy(1); energy_ratio(1)]; % Initial state

% UKF function
[filtered_total_energy, filtered_energy_ratio] = unscented_kalman_filter(total_energy, energy_ratio, Q, R, P, x);

% Calculate the number of data points for each quarter
quarter_length = floor(length(time) / 2);

for i = 1:2
    % Calculate start and end indices for each quarter
    start_idx = (i-1) * quarter_length + 1;
    end_idx = min(i * quarter_length, length(time));
    
    % Extract data for the current quarter
    quarter_time = time(start_idx:end_idx);
    quarter_total_energy = filtered_total_energy(start_idx:end_idx);
    quarter_energy_ratio = filtered_energy_ratio(start_idx:end_idx);
    
    % Create new figure window, adjust height and width
    figure('Position', [100, 100, 1200, 800]);  % Increase height to provide more space

    % Add overall title
    annotation('textbox', [0, 0.96, 1, 0.04], ...
               'String', sprintf('Quarter %d: %s to %s', i, datestr(quarter_time(1), 'HH:MM:SS'), datestr(quarter_time(end), 'HH:MM:SS')), ...
               'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
               'VerticalAlignment', 'top', 'FontSize', 10, 'FontWeight', 'bold');

    % Plot energy ratio
    ax1 = subplot(2,1,1);
    plot(quarter_time, quarter_energy_ratio, '-', 'LineWidth', 1, 'Color', [.3 .3 .3]);
    ylabel('High/Low');
    title('Energy Ratio Over Time');
    grid on;

    % Plot total energy (log scale)
    ax2 = subplot(2,1,2);
    semilogy(quarter_time, quarter_total_energy, '-', 'LineWidth', 1, 'Color', [.3 .3 .3]);
    ylabel('Total Energy (log scale)');
    title('Total Energy Over Time');
    grid on;

    % Set x-axis time format for both subplots
    for j = 1:2
        ax = subplot(2,1,j);
        ax.XAxis.TickValues = quarter_time(1):minutes(30):quarter_time(end); % Tick every 30 minutes
        ax.XAxis.TickLabelFormat = 'HH:mm';
        xtickangle(45); % Rotate x-axis labels to prevent overlap

        % Adjust graph aesthetics
        set(ax, 'FontName', 'Arial', 'FontSize', 10, 'LineWidth', 1);
        box on;

        % Display x-axis labels on both subplots
        xlabel('Time', 'FontWeight', 'bold');
    end

    % Adjust subplot positions to eliminate overlap
    pos1 = get(ax1, 'Position');
    pos2 = get(ax2, 'Position');
    set(ax1, 'Position', [pos1(1), pos1(2)+0.08, pos1(3), pos1(4)-0.15]);
    set(ax2, 'Position', [pos2(1), pos2(2)+0.08, pos2(3), pos2(4)-0.15]);

    % Adjust overall figure layout
    set(gcf, 'Color', 'w');

    % Save figure
    filename = sprintf('Quarter_%d_Energy_Plot_Filtered.png', i);
    saveas(gcf, filename);

end

disp('All figures have been saved.');

% UKF function definition
function [filtered_total_energy, filtered_energy_ratio] = unscented_kalman_filter(total_energy, energy_ratio, Q, R, P, x)
    n = length(x);
    m = length(R);
    N = length(total_energy);
    
    filtered_total_energy = zeros(N, 1);
    filtered_energy_ratio = zeros(N, 1);
    
    for k = 1:N
        % Prediction
        [X, W] = calculate_sigma_points(x, P);
        [x_pred, P_pred] = predict_state_covariance(X, W, Q);
        
        % Update
        z = [total_energy(k); energy_ratio(k)];
        [Z, W] = calculate_sigma_points(x_pred, P_pred);
        [z_pred, Pzz] = predict_measurement(Z, W, R);
        
        Pxz = calculate_cross_covariance(X, Z, x_pred, z_pred, W);
        
        K = Pxz / Pzz;
        x = x_pred + K * (z - z_pred);
        P = P_pred - K * Pzz * K';
        
        filtered_total_energy(k) = x(1);
        filtered_energy_ratio(k) = x(2);
    end
end

% Helper functions for UKF
function [X, W] = calculate_sigma_points(x, P)
    n = length(x);
    alpha = 1e-3;
    kappa = 0;
    beta = 2;
    lambda = alpha^2 * (n + kappa) - n;
    
    X = zeros(n, 2*n+1);
    W = zeros(1, 2*n+1);
    
    X(:,1) = x;
    W(1) = lambda / (n + lambda);
    
    sqrtP = chol((n + lambda) * P)';
    for i = 1:n
        X(:,i+1) = x + sqrtP(:,i);
        X(:,i+1+n) = x - sqrtP(:,i);
        W(i+1) = 1 / (2*(n + lambda));
        W(i+1+n) = 1 / (2*(n + lambda));
    end
end

function [x_pred, P_pred] = predict_state_covariance(X, W, Q)
    [n, L] = size(X);
    x_pred = zeros(n, 1);
    P_pred = zeros(n, n);
    
    for i = 1:L
        x_pred = x_pred + W(i) * X(:,i);
    end
    
    for i = 1:L
        diff = X(:,i) - x_pred;
        P_pred = P_pred + W(i) * (diff * diff');
    end
    
    P_pred = P_pred + Q;
end

function [z_pred, Pzz] = predict_measurement(Z, W, R)
    [m, L] = size(Z);
    z_pred = zeros(m, 1);
    Pzz = zeros(m, m);
    
    for i = 1:L
        z_pred = z_pred + W(i) * Z(:,i);
    end
    
    for i = 1:L
        diff = Z(:,i) - z_pred;
        Pzz = Pzz + W(i) * (diff * diff');
    end
    
    Pzz = Pzz + R;
end

function Pxz = calculate_cross_covariance(X, Z, x_pred, z_pred, W)
    [n, L] = size(X);
    m = size(Z, 1);
    Pxz = zeros(n, m);
    
    for i = 1:L
        dx = X(:,i) - x_pred;
        dz = Z(:,i) - z_pred;
        Pxz = Pxz + W(i) * (dx * dz');
    end
end
