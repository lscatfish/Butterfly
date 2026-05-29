%% 1. 已知参数（固定圆半径l可按需调整）
clear; clc; close all;  % 清理工作区，避免旧变量干扰
a = 8;    
b = 6.97;    
R = 2.25;    
c = 14;
l = 8;    % 固定圆 x²+y²=l² 的半径

%% 2. 角度与时间参数（时间步长物理意义对齐）
theta_deg = linspace(0, 360, 360);  % 0-360度，共360帧
theta_rad = deg2rad(theta_deg);
frame_interval = 0.05;  % 动画帧间隔（单位：秒），和pause参数完全对应
dt = frame_interval;     % 正确的时间步长，物理意义：每帧耗时0.05s

%% 3. 预分配数组（初始化为NaN，避免错误0值污染结果）
n_frame = length(theta_rad);
x1 = zeros(1, n_frame);
y1 = zeros(1, n_frame);
x2 = zeros(1, n_frame);
y2 = zeros(1, n_frame);
x_intersect = NaN(1, n_frame);  % 交点x坐标，无交点时为NaN
y_intersect = NaN(1, n_frame);  % 交点y坐标，无交点时为NaN
phi = NaN(1, n_frame);          % 交点极角（rad），先存原始值，后续解缠绕

%% 4. 动画初始化
figure('Color','w', 'Position', [100, 100, 800, 600]);
% 主点、跟随点、交点的动画线
h1 = animatedline('Marker','o','MarkerSize',8,'Color','b','DisplayName','主点','MaximumNumPoints',1);
h2 = animatedline('Marker','*','MarkerSize',10,'Color','r','DisplayName','跟随点','MaximumNumPoints',1);
h_intersect = animatedline('Marker','s','MarkerSize',7,'Color','m','DisplayName','直线-圆交点','MaximumNumPoints',1);
h_intersect_trace = animatedline('Color','m','LineStyle',':','LineWidth',1,'DisplayName','交点轨迹'); % 新增交点轨迹
hold on; grid on; axis equal;
xlabel('x 坐标 (m)'); ylabel('y 坐标 (m)');
legend('Location','best');

% 绘制固定圆 x²+y²=l²
circle_theta = linspace(0, 2*pi, 360);  
circle_x = l * cos(circle_theta);      
circle_y = l * sin(circle_theta);      
plot(circle_x, circle_y, 'g--', 'LineWidth', 1.2, 'DisplayName', ['固定圆 x²+y²=', num2str(l^2)]);

% 动态直线初始化
line_h = plot(NaN, NaN, 'k-', 'LineWidth', 1.5, 'DisplayName', '动态直线'); 

%% 5. 循环计算：先求交点和极角，循环后统一计算角加速度
for i = 1:n_frame
    theta = theta_rad(i);
    
    % --- 主点位置（沿 (x-a)²+(y-b)²=R² 圆周运动）
    x1(i) = a + R*cos(theta);
    y1(i) = b + R*sin(theta);
    
    % --- 动态直线方程系数
    A = x1(i);
    B = y1(i);
    C = (x1(i)^2 + y1(i)^2 - c)/2;
    
    % --- 跟随点位置
    t = i/50;  
    if abs(B) > 1e-10
        x2(i) = t;
        y2(i) = (C - A*x2(i)) / B;
    else
        x2(i) = C/A;
        y2(i) = t;
    end
    
    % ================== 直线与固定圆交点求解 ==================
    % 联立方程：A*x + B*y = C  &&  x² + y² = l²
    valid_intersect = false;
    if abs(A) < 1e-10 && abs(B) < 1e-10
        % 极端情况：直线退化，无意义，跳过
        x_intersect(i) = NaN;
        y_intersect(i) = NaN;
    elseif abs(B) > 1e-10
        % 情况1：B≠0，消去y，解一元二次方程
        coeff = [A^2 + B^2,  -2*A*C,  C^2 - B^2*l^2];
        roots_x = roots(coeff);
        % 过滤虚数解，只保留实数交点
        real_idx = abs(imag(roots_x)) < 1e-10;
        real_x = real(roots_x(real_idx));
        real_y = (C - A*real_x) / B;
    else
        % 情况2：B=0，直线为垂直直线 A*x=C，消去x，解y
        x0 = C/A;
        y_sq = l^2 - x0^2;
        if y_sq >= -1e-10  % 允许微小负数值（浮点误差），对应相切/有交点
            y_sq = max(y_sq, 0);
            real_y = [-sqrt(y_sq), sqrt(y_sq)];
            real_x = x0 * ones(size(real_y));
        else
            real_x = [];
            real_y = [];
        end
    end

    % ================== 【已固定启用选项2】单交点选取：始终取x坐标最大的交点 ==================
    if ~isempty(real_x)
        % 核心逻辑：取x坐标最大的交点，固定观察圆右侧区域
        [~, select_idx] = max(real_x);  

        % 执行选取
        x_intersect(i) = real_x(select_idx);
        y_intersect(i) = real_y(select_idx);
        valid_intersect = true;
    else
        % 无交点，保持NaN
        x_intersect(i) = NaN;
        y_intersect(i) = NaN;
    end

    % --- 计算交点极角（atan2自动处理象限，后续统一解缠绕）
    if valid_intersect
        phi(i) = atan2(y_intersect(i), x_intersect(i));
    end

    % ================== 动画更新 ==================
    % 更新动态直线（扩大范围，确保直线完整显示）
    if abs(B) > 1e-10
        x_line = [-l-10, l+10];  
        y_line = (C - A*x_line) / B;
    else
        y_line = [-l-10, l+10];  
        x_line = (C - B*y_line) / A;
    end
    set(line_h, 'XData', x_line, 'YData', y_line);
    
    % 更新所有点的位置
    clearpoints(h1); addpoints(h1, x1(i), y1(i));
    clearpoints(h2); addpoints(h2, x2(i), y2(i));
    if valid_intersect
        clearpoints(h_intersect); addpoints(h_intersect, x_intersect(i), y_intersect(i));
        addpoints(h_intersect_trace, x_intersect(i), y_intersect(i));
    end
    
    % 坐标轴自适应，保证所有元素在视野内
    axis_lim = l + max([a+R, b+R, 10]);
    xlim([-axis_lim, axis_lim]);
    ylim([-axis_lim, axis_lim]);
    
    % 实时更新标题，显示当前核心参数
    title(sprintf('帧号：%d/360 | 交点角加速度：%.4f rad/s² | 角速度：%.4f rad/s', ...
        i, 0, 0), 'FontSize',11);
    
    drawnow limitrate;  % 优化动画流畅度
    pause(frame_interval);   
end

%% 6. 统一计算角速度、角加速度（先解缠绕，再微分）
% --- 极角解缠绕：消除atan2带来的[-π, π]跳变，解决微分爆炸问题
phi_unwrap = unwrap(phi);

% --- 计算角速度（极角的一阶导数）
omega = gradient(phi_unwrap, dt);  

% --- 计算角加速度（角速度的一阶导数 = 极角的二阶导数）
alpha = gradient(omega, dt);  

%% 7. 结果可视化与验证
% --- 实时更新动画标题的最终数值
for i = 1:n_frame
    if ~isnan(alpha(i))
        title_str = sprintf('帧号：%d/360 | 交点角加速度：%.4f rad/s² | 角速度：%.4f rad/s', ...
            i, alpha(i), omega(i));
        set(gcf, 'CurrentAxes', gca);
        title(title_str, 'FontSize',11);
        drawnow limitrate;
        pause(0.01);
    end
end

% --- 绘制角加速度、角速度随时间变化的曲线
time = (0:n_frame-1) * dt;  % 时间轴和帧完全对应
figure('Color','w', 'Position', [150, 150, 900, 600]);

% 子图1：角速度
subplot(2,1,1);
plot(time, omega, 'b-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('角速度 ω (rad/s)');
title('交点角速度随时间变化曲线');
grid on;

% 子图2：角加速度
subplot(2,1,2);
plot(time, alpha, 'r-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('角加速度 α (rad/s²)');
title('交点角加速度随时间变化曲线');
grid on;

% --- 输出关键结果到工作区
fprintf('计算完成！已固定选取【x坐标最大的交点】\n');
fprintf('最大角加速度：%.4f rad/s²\n', max(alpha, [], 'omitnan'));
fprintf('最小角加速度：%.4f rad/s²\n', min(alpha, [], 'omitnan'));
fprintf('平均角加速度：%.4f rad/s²\n', mean(alpha, 'omitnan'));