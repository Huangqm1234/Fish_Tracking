import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import savgol_filter

def read_trace_file(file_path):
    trace_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_id = None
        current_trace = []
        for line in lines:
            line = line.strip()
            if line.startswith('**'):
                if current_id is not None:
                    trace_data[current_id] = current_trace
                    current_trace = []
                current_id = int(line.split('=')[1].strip()[0])
            elif line == '-' * 10:
                if current_id is not None:
                    trace_data[current_id] = current_trace
                    current_id = None
                    current_trace = []
            else:
                if current_id is not None:
                    x, y = map(float, line.split(','))
                    current_trace.append([x, y])
    return trace_data

# 使用示例
file_path = 'outputs\\trace_1_tran.txt'
trace_data = read_trace_file(file_path)
#print(trace_data[1][950])

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def transform_coordinates(x0, y0, x1, y1, x2, y2):
    '''
    x0, y0: 起始点坐标
    x1, y1: 待变换点坐标
    x2, y2: 新y轴指向
    '''
    # 计算向量 v 和其长度 d
    dx = x2 - x0
    dy = y2 - y0
    d = math.sqrt(dx**2 + dy**2)

    # 计算单位向量 u 和垂直向量 u_perp
    ux = dx / d
    uy = dy / d
    u_perp_x = -uy
    u_perp_y = ux

    # 平移坐标
    x1_prime = x1 - x0
    y1_prime = y1 - y0

    # 计算新坐标
    x1_new = x1_prime * u_perp_x + y1_prime * u_perp_y
    y1_new = x1_prime * ux + y1_prime * uy

    return x1_new, y1_new

def smooth_trace(relative_location, window_size=50, polyorder=5):
    '''
    对追踪结果进行平滑处理
    '''
    smooth_relative_location = []
    x_values = [point[0] for point in relative_location]
    y_values = [point[1] for point in relative_location]
    smoothed_x = savgol_filter(x_values, window_length=window_size, polyorder = polyorder)
    smoothed_y = savgol_filter(y_values, window_length=window_size, polyorder = polyorder)
    for i in range(len(x_values)):
        smooth_relative_location.append([smoothed_x[i], smoothed_y[i]])
    
    return smooth_relative_location


def draw_relative_trace(trace_data):
    point_numbers = len(trace_data[0]) # 获得点的数量

    relative_location = [] # 记录相对坐标

    for i in range(point_numbers):
        x0, y0 = trace_data[0][i]
        x1, y1 = trace_data[1][i]
        x2, y2 = trace_data[2][i]

        #y_slope = (y2-y0)/(x2-x0) # 新y斜率
        #y_angle = math.atan(y_slope) # 新y与x轴夹角弧度
        #
        #y_rel_slope = (y1-y0)/(x1-x0) 
        #y_rel_angle = math.atan(y_rel_slope)

        #angle = y_rel_angle-y_angle
        #dis = distance(x0, y0, x1, y1)
        #x1_relative = dis * math.sin(angle)
        #y1_relative = dis * math.cos(angle)

        # 坐标变换
        x1_relative, y1_relative = transform_coordinates(x0, y0, x1, y1, x2, y2)


        relative_location.append([(x1_relative * 1.0)/400, (y1_relative * 1.0)/400])

    smooth_relative_location = smooth_trace(relative_location)
    return smooth_relative_location, relative_location

smooth_relative_location,relative_location = draw_relative_trace(trace_data)


plt.plot(relative_location[0][0], relative_location[0][1], 'ro')  # 绘制红色起点

# 绘制relative_location中的点
#for i in range(len(relative_location) - 1):
#    plt.plot([relative_location[i][0], relative_location[i + 1][0]], [relative_location[i][1], relative_location[i + 1][1]], 'b-')  # 绘制蓝色实线

# 绘制smooth_relative_location中的点
for i in range(len(smooth_relative_location) - 1):
    plt.plot([smooth_relative_location[i][0], smooth_relative_location[i + 1][0]], [smooth_relative_location[i][1], smooth_relative_location[i + 1][1]], 'r-')  # 绘制红色实线

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Relative Location of Points')
plt.grid(True)
plt.show()



