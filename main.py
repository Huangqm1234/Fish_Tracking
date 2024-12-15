import cv2
import mmcv
import tempfile
import numpy as np
from mmtrack.apis import inference_sot, init_model

import seaborn as sns
import random
# 生成调色板
palette = sns.color_palette('hls', 20)

def get_color(seed):
    '''
    传入追踪ID，生成专属颜色
    '''
    random.seed(seed)
    # 从调色板中随机挑选一种颜色
    bbox_color = random.choice(palette)
    bbox_color = [int(255 * c) for c in bbox_color][::-1]
    return bbox_color

def save_trace(circle_coord_list: list, output_file: str, ID_num: int):
    '''
    保存追踪结果

    参数:
        circle_coord_list (list): 包含追踪坐标的列表字典，其中ID键的trace键为追踪坐标列表
        output_file (str): 输出文件的路径
        ID_num (int): 追踪目标的数量

    返回:
        None
    '''
    with open(output_file, 'w') as f:
        for ID in range(ID_num):
            f.write('**     ID = {}     **\n'.format(ID))
            for each in circle_coord_list[ID]['trace']:
                f.write('{},{}\n'.format(each[0], each[1]))
            f.write('-'*10)
            f.write('\n')


def main():

    # 文件名称
    file_name = '1_tran'

    # 输入输出视频路径
    input_video = 'data\\' + file_name + '.mp4'
    output = 'outputs\\output_' + file_name + '.mp4'
    trace_ouput = 'outputs\\trace_' + file_name + '.txt'

    # 指定单目标追踪算法 config 配置文件
    sot_config = 'siamese_rpn_r50_20e_lasot.py'
    # 指定单目标检测算法的模型权重文件
    sot_checkpoint = 'https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth'
    # 初始化单目标追踪模型
    sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')

    # 指定多个目标的初始矩形框坐标 [x, y, w, h]
    init_bbox_xywh = [[280, 572, 9, 11], [501, 499, 8, 11],[316, 565, 11, 11]]

    # 目标个数
    ID_num=len(init_bbox_xywh)
    print('共有{}个待追踪目标'.format(ID_num))

    # 转成 [x1, y1, x2, y2 ]
    init_bbox_xyxy = []
    for each in init_bbox_xywh:
        init_bbox_xyxy.append([each[0], each[1], each[0]+each[2], each[1]+each[3]])
    
    # 读入待预测视频
    imgs = mmcv.VideoReader(input_video)
    # prog_bar = mmcv.ProgressBar(len(imgs))
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name

    ## 获取每帧的追踪结果
    # 逐帧输入模型预测
    circle_coord_list = {}
    print('开始逐帧处理')

    for ID in range(ID_num): # 遍历每个待追踪目标
        print('\n')
        print('追踪第{}个目标'.format(ID+1))
        circle_coord_list[ID] = {}
        circle_coord_list[ID]['bbox'] = [] # 保存框的坐标
        circle_coord_list[ID]['trace'] = [] # 保存跟踪中心点的坐标

        # 启动进度条
        prog_bar = mmcv.ProgressBar(len(imgs))

        for i, img in enumerate(imgs): # 遍历视频每一帧

            # 执行单目标追踪
            result = inference_sot(sot_model, img, init_bbox_xyxy[ID], frame_id=i)
            # 目标检测矩形框坐标
            result_bbox = np.array(result['track_bboxes'][:4].astype('uint32'))
            # 保存矩形框坐标
            circle_coord_list[ID]['bbox'].append(result_bbox)


            # 获取矩形框中心点轨迹点坐标
            circle_x = int((result_bbox[0] + result_bbox[2]) / 2)
            circle_y = int((result_bbox[1] + result_bbox[3]) / 2)
            # 保存轨迹点坐标
            circle_coord_list[ID]['trace'].append(np.array([circle_x, circle_y]))

            prog_bar.update()

    ## 可视化
    # 启动进度条
    prog_bar = mmcv.ProgressBar(len(imgs))
    
    for i, img in enumerate(imgs): # 遍历视频每一帧
        img_draw = img.copy()
        
        for ID in range(ID_num): # 遍历每个待追踪目标
            # 获取该目标的专属颜色
            ID_color = get_color(ID)
            
            result_bbox = circle_coord_list[ID]['bbox'][i]
            
            # 绘制目标检测矩形框：图像，左上角坐标，右下角坐标，颜色，线宽
            img_draw = cv2.rectangle(img_draw, (result_bbox[0], result_bbox[1]), (result_bbox[2], result_bbox[3]), ID_color, 2)  
    
            # 绘制从第一帧到当前帧的轨迹
            for each in circle_coord_list[ID]['trace'][:i]:
                # 绘制圆，指定圆心坐标和半径，红色，最后一个参数为线宽，-1表示填充
                img_draw = cv2.circle(img_draw, (each[0],each[1]), 2,  ID_color, -1)
        
        # 将当前帧的可视化效果保存为图片文件
        cv2.imwrite(f'{out_path}/{i:06d}.jpg', img_draw)
        prog_bar.update()
        
    # 将保存下来的各帧图片文件串成视频
    print('导出视频，FPS {}'.format(imgs.fps))
    mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
    print('已成功导出视频 至 {}'.format(output))
    out_dir.cleanup()

    # 保存追踪结果
    save_trace(circle_coord_list, trace_ouput, ID_num)


if __name__ == '__main__':
    main()