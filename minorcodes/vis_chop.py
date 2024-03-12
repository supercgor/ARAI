import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate(points, rotation_vector):
    """
    Rotate the points with rotation_vector.

    Args:
        points (_type_): shape (N, 3)
        rotation_vector (_type_): rotation along x, y, z axis. shape (3,)
    """
    if points.shape[1] == 3:
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        return np.dot(points, rotation_matrix.T)
    
    elif points.shape[1] == 2:
        rotation_vector = np.array([0, 0, rotation_vector])
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        return np.dot(points, rotation_matrix.T[:2, :2])

def is_inside_rectangle(x, y, rect_points):
    """ 检查点是否在旋转后的长方形内 """
    # 使用向量叉乘的方法检查点是否在所有边的同一侧
    for i in range(4):
        p0, p1 = rect_points[i], rect_points[(i + 1) % 4]
        vec1 = (p1[0] - p0[0], p1[1] - p0[1])
        vec2 = (x - p0[0], y - p0[1])
        if np.cross(vec1, vec2) < 0:
            return False
    return True

def rectangle_points(S):
    """ 计算长方形的角点 """
    return np.array([(-S[0]/2, -S[1]/2), (S[0]/2, -S[1]/2), (S[0]/2, S[1]/2), (-S[0]/2, S[1]/2)])

def find_square_centers(S, M, k, theta):
    """ 寻找所有正方形的中心坐标 """
    # 计算旋转后长方形的角点
    rect_points = rotate(rectangle_points(S), np.deg2rad(theta))

    centers = []
    # 遍历可能的正方形中心点
    r = max(S)
    for x in np.arange(-r, r, k[0]):
        for y in np.arange(-r, r, k[1]):
            # 检查正方形的角点是否都在长方形内
            if all(is_inside_rectangle(x + dx, y + dy, rect_points)
                   for dx in [-M/2, M/2] for dy in [-M/2, M/2]):
                centers.append((x, y))

    return centers

# 参数示例
S = np.array([10, 7])       # 长方形的尺寸
M = 1                       # 正方形的边长
k = np.array([0.5, 0.5])    # 步长
theta = 70                  # 旋转角度

# 计算正方形中心
centers = find_square_centers(S, M, k, theta)
print("正方形中心坐标:", centers)

# 绘图
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 绘制旋转后的长方形
def draw_rotated_rectangle(S, theta, ax):
    rect = patches.Rectangle((-S[0]/2, -S[1]/2), S[0], S[1], angle=theta, fill=False, edgecolor='red', lw=4, rotation_point='center')
    ax.add_patch(rect)

# 绘制正方形
def draw_squares(centers, M, ax):
    for x, y in centers:
        square = patches.Rectangle((x - M/2, y - M/2), M, M, fill=False, edgecolor='blue', lw=2)
        ax.add_patch(square)

# 绘图设置
fig, ax = plt.subplots(figsize=(8, 8))
r = np.ceil(max(S) / np.sqrt(2))

ax.set_xticks(np.arange(-r, r+1, 1))
ax.set_yticks(np.arange(-r, r+1, 1))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-r, r)
ax.set_ylim(-r, r)
ax.grid()
draw_rotated_rectangle(S, theta, ax)
draw_squares(centers, M, ax)
plt.show()