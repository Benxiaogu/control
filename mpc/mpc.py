import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --------------------- MPC_Matrices Function ---------------------
def mpc_matrices(A, B, Q, R, F, N, target):
    """
        A:系统状态
        B:输入矩阵
        Q,R,F:加权矩阵，用于目标函数
        N:预测区间长度
    """
    n = A.shape[0]  # 状态空间维度
    p = B.shape[1]  # 输入的维度

    M = np.zeros(((N + 1) * n, n)) # 用来存储未来预测的状态
    M[:n, :] = np.eye(n)
    C = np.zeros(((N + 1) * n, N * p)) # 用于预测未来的输入

    T = np.zeros(((N+1)*n , n)) # 用来存储未来的参考
    T[n:2*n,:] = A - np.eye(n)

    tmp = np.eye(n) # 用于记录A^i的累积乘积

    # 更新M和C
    for i in range(1, N + 1):
        rows = slice(i * n, (i + 1) * n)
        prev_C = C[rows.start - n:rows.start, :i * p - p] if (i * p - p) > 0 else np.zeros((n, 0))
        C[rows, :i * p] = np.hstack([tmp @ B, prev_C])
        tmp = A @ tmp
        M[rows, :] = tmp

        if i == 1:
            continue

        T[i*n:(i+1)*n] = A**i @ (A - np.eye(n))



    Q_bar = np.kron(np.eye(N), Q)
    Q_bar = np.block([[Q_bar, np.zeros((N * n, n))], [np.zeros((n, N * n)), F]])
    R_bar = np.kron(np.eye(N), R)

    G = M.T @ Q_bar @ M
    E = M.T @ Q_bar @ C
    H = C.T @ Q_bar @ C + R_bar

    W = target.T @ T.T @ Q_bar @ C
    Y = M.T @ Q_bar @ T @ target
    Z = target.T @ T.T @ Q_bar @ T @ target


    return G, E, H, W, Y, Z

# --------------------- Prediction Function ---------------------
def prediction(x_k, G, E, H, W, Y, Z, N, p):
    """
        x_k:当前状态
        E,H:从mpc_matrices()得到的矩阵
        N,p:预测区间长度和输入维度
    """

    # 定义目标函数
    def objective(U_k):
        # return 0.5 * U_k.T @ H @ U_k + x_k.T @ E @ U_k + W @ U_k + 0.5 * x_k.T @ G @ x_k + x_k.T @ Y + 0.5 * Z
        return 0.5 * U_k.T @ H @ U_k + x_k.T @ E @ U_k + W @ U_k

    U_k0 = np.zeros(N * p)
    bounds = [(None, None)] * (N * p)

    # 求解二次规划
    result = minimize(objective, U_k0, bounds=bounds, method='trust-constr', options={'disp': False}) 
    # trust-constr 是 scipy.optimize.minimize 提供的一种优化算法，用于解决带有约束的非线性优化问题。
    # 其全称为“Trust Region Constrained Algorithm”，它结合了信赖域（Trust Region）和约束优化的概念

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    U_k = result.x
    return U_k[:p] 

# --------------------- Main Program ---------------------
A = np.array([[1, 0.1], [-1, 2]])
B = np.array([[0.2, 1], [0.5, 2]])
Q = np.array([[100, 0], [0, 10]])
F = np.array([[100, 0], [0, 1]])
R = np.array([[0.1, 0], [0, 0.1]])

k_steps = 200 # 仿真总步数
N = 5 # 预测区间长度
n = A.shape[0] # 状态空间的维度
p = B.shape[1] # 输入空间的维度

# 初始化状态和输入矩阵
X_K = np.zeros((n, k_steps + 1))
U_K = np.zeros((p, k_steps))
X_K[:, 0] = np.array([20, -20]) # 系统状态初始化

target = np.array([10, 10])  # 系统参考值

# 计算E和H矩阵
G, E, H, W, Y, Z = mpc_matrices(A, B, Q, R, F, N, target)


for k in range(k_steps):
    try:
        U_K[:, k] = prediction(X_K[:, k], G, E, H, W, Y, Z, N, p)
    except ValueError as e:
        print(f"Step {k}: {e}")
        break
    X_K[:, k + 1] = A @ X_K[:, k] + B @ U_K[:, k]

# 绘制结果
plt.figure(figsize=(10, 8))

# 绘制状态变量
plt.subplot(2, 1, 1)
for i in range(n):
    plt.plot(X_K[i, :], label=f'x{i + 1}')
plt.title('State Variables Over Time')
plt.legend()
plt.grid()

# 绘制系统输入量
plt.subplot(2, 1, 2)
for i in range(p):
    plt.plot(U_K[i, :], label=f'u{i + 1}')
plt.title('Control Inputs Over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
