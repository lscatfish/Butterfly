%% 1. 核心参数设置
clear; clc; close all;

% --- 固定参数 ---
b = 6;    
R = 2.25;    
c = 14;
l = 8;    

% --- 扫描参数：a值从6到13变化 ---
a_start = 6;
a_end = 13;
n_a = 20;  % a值的扫描点数（越多曲面越平滑，建议15-30）
a_vals = linspace(a_start, a_end, n_a);  % 生成a值序列

% --- 时间/角度参数 ---
theta_deg = linspace(0, 360, 360);  % 0-360度，共360帧
theta_rad = deg2rad(theta_deg);
frame_interval = 0.05;  % 时间步长（仅用于物理意义，无动画）
dt = frame_interval;     
n_frame = length(theta_rad);

%% 2. 预分配二维数组（行：时间/帧，列：a值）
omega_matrix = NaN(n_frame, n_a);  % 角速度矩阵：omega(t, a)
alpha_matrix = NaN(n_frame, n_a);  % 角加速度矩阵（可选）

%% 3. 外层循环：扫描a值（6到13）
fprintf('开始计算，共 %d 个a值，请稍候...\n', n_a);
for a_idx = 1:n_a
    a = a_vals(a_idx);  % 当前a值
    
    % --- 预分配当前a值的一维数组 ---
    x1 = zeros(1, n_frame);
    y1 = zeros(1, n_frame);
    x_intersect = NaN(1, n_frame);
    y_intersect = NaN(1, n_frame);
    phi = NaN(1, n_frame);
    
    % --- 内层循环：遍历角度theta（0-360度）---
    for i = 1:n_frame
        theta = theta_rad(i);
        
        % 1. 主点位置
        x1(i) = a + R*cos(theta);
        y1(i) = b + R*sin(theta);
        
        % 2. 动态直线方程系数
        A = x1(i);
        B = y1(i);
        C = (x1(i)^2 + y1(i)^2 - c)/2;
        
        % 3. 直线与固定圆交点求解
        valid_intersect = false;
        if abs(A) < 1e-10 && abs(B) < 1e-10
            x_intersect(i) = NaN; y_intersect(i) = NaN;
        elseif abs(B) > 1e-10
            coeff = [A^2 + B^2,  -2*A*C,  C^2 - B^2*l^2];
            roots_x = roots(coeff);
            real_idx = abs(imag(roots_x)) < 1e-10;
            real_x = real(roots_x(real_idx));
            real_y = (C - A*real_x) / B;
        else
            x0 = C/A;
            y_sq = l^2 - x0^2;
            if y_sq >= -1e-10
                y_sq = max(y_sq, 0);
                real_y = [-sqrt(y_sq), sqrt(y_sq)];
                real_x = x0 * ones(size(real_y));
            else
                real_x = []; real_y = [];
            end
        end
        
        % 4. 单交点选取：始终取x坐标最大的交点
        if ~isempty(real_x)
            [~, select_idx] = max(real_x);
            x_intersect(i) = real_x(select_idx);
            y_intersect(i) = real_y(select_idx);
            valid_intersect = true;
        end
        
        % 5. 计算交点极角
        if valid_intersect
            phi(i) = atan2(y_intersect(i), x_intersect(i));
        end
    end
    
    % --- 计算当前a值下的角速度和角加速度 ---
    phi_unwrap = unwrap(phi);  % 极角解缠绕
    omega = gradient(phi_unwrap, dt);  % 角速度
    alpha = gradient(omega, dt);        % 角加速度（可选）
    
    % --- 存入二维矩阵 ---
    omega_matrix(:, a_idx) = omega;
    alpha_matrix(:, a_idx) = alpha;
    
    fprintf('已完成 a = %.2f (%d/%d)\n', a, a_idx, n_a);
end
fprintf('所有计算完成！\n');

%% 4. 绘制三维曲面图（核心结果）
time = (0:n_frame-1) * dt;  % 时间轴
[T, A] = meshgrid(time, a_vals);  % 生成网格坐标

figure('Color','w', 'Position', [100, 100, 1000, 700]);

% --- 子图1：角速度 omega(t, a) 三维曲面图 ---
subplot(2,1,1);
surf(T, A, omega_matrix');  % 注意转置：omega_matrix是(t,a)，surf需要(a,t)
shading interp;  % 平滑着色
xlabel('时间 t (s)');
ylabel('参数 a');
zlabel('角速度 ω (rad/s)');
title('角速度 ω 随 a 值(6-13)和时间 t 的变化曲面');
colorbar;  % 显示颜色条
view(3);  % 三维视角
grid on;

% --- 子图2：角速度的等高线图（辅助分析）---
subplot(2,1,2);
contourf(T, A, omega_matrix', 20, 'LineColor','none');  % 填充等高线
xlabel('时间 t (s)');
ylabel('参数 a');
colorbar;
title('角速度 ω 的等高线图（颜色越深表示角速度绝对值越大）');
grid on;
