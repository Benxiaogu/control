import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 系统参数 (A 和 B 矩阵)
A = np.array([[1, 0.1], [-1, 2]])  # 状态转移矩阵
B = np.array([[0.2, 1], [0.5, 2]])  # 控制输入矩阵

# MPC参数
Q = np.array([[100, 0], [0, 1]])  # 状态误差的权重矩阵
P = np.array([[100, 0], [0, 1]])  # 终端状态的权重矩阵
R = np.array([[0.1, 0], [0, 0.1]])  # 控制输入的权重矩阵

N = 5  # 预测区间长度
k_steps = 200  # 仿真总步数

# 初始状态和参考目标状态
x0 = np.array([5, 0])  # 初始状态
x_ref = np.array([0, 0])  # 参考状态（目标状态）

# 定义优化变量
x = cp.Variable((2, N+1))  # 状态变量，大小为(2, N+1)
u = cp.Variable((2, N))    # 控制输入变量，大小为(2, N)

# 代价函数
cost = 0
for k in range(N):
    cost += cp.quad_form(x[:, k] - x_ref, Q) + cp.quad_form(u[:, k], R)

# 终端状态代价
cost += cp.quad_form(x[:, N] - x_ref, P)

# 系统动态约束
constraints = []
for k in range(N):
    if k == 0:
        constraints += [x[:, k] == x0]  # 初始状态
    else:
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k-1]]  # 状态方程

# 执行MPC优化
def mpc_control(x0):
    # 设置初始状态
    constraints[0] = x[:, 0] == x0
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective, constraints)
    
    # 使用ECOS求解器（可以替换为SCS求解器）
    problem.solve(solver=cp.ECOS, verbose=True)
    
    # 返回最优控制输入
    return u.value[:, 0]

# 仿真
x_values = np.zeros((2, k_steps))  # 状态轨迹
u_values = np.zeros((2, k_steps))  # 控制输入轨迹

x_values[:, 0] = x0  # 初始化状态
for t in range(k_steps - 1):
    u_opt = mpc_control(x_values[:, t])  # 计算最优控制输入
    u_values[:, t] = u_opt  # 记录控制输入
    x_values[:, t + 1] = A @ x_values[:, t] + B @ u_opt  # 更新状态

# 绘制结果
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 状态轨迹
ax1.plot(x_values[0, :], label="x1 (Position)", color="b")
ax1.plot(x_values[1, :], label="x2 (Velocity)", color="r")
ax1.set_title("State Trajectory")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("State")
ax1.legend()
ax1.grid()

# 控制输入
ax2.plot(u_values[0, :], label="u1", color="g")
ax2.plot(u_values[1, :], label="u2", color="orange")
ax2.set_title("Control Inputs")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Control Input")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
