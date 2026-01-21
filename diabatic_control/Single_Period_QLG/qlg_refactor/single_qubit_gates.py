"""Common single-qubit gates used by the notebook."""

from __future__ import annotations

import numpy as np

# 基础矩阵
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
# ---- 单比特 Clifford+T 基本门 ----
T = np.array([[1, 0],
              [0, np.exp(1j*np.pi/4)]], dtype=complex)  # = Rz(pi/4)
# 旋转门
def Rx(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*X

def Ry(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y

def Rz(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Z

# 所有 24 个 Clifford 元素
cliffords = [
    I,                          # 1
    X, Y, Z,                    # 2,3,4
    H, H@S, H@S@S, H@S@S@S,     # 5,6,7,8
    S, S@S, S@S@S,              # 9,10,11
    Rx(np.pi/2), Rx(-np.pi/2),  # 12,13
    Ry(np.pi/2), Ry(-np.pi/2),  # 14,15
    Rz(np.pi/2), Rz(-np.pi/2),  # 16,17
    H@Rx(np.pi/2), H@Rx(-np.pi/2), # 18,19
    H@Ry(np.pi/2), H@Ry(-np.pi/2), # 20,21
    H@Rz(np.pi/2), H@Rz(-np.pi/2)  # 22,23
]

# 最后一个 Clifford = -I （模掉相位等价于 I，但通常也计入 24 个之一）
cliffords.append(-I)  # 24
