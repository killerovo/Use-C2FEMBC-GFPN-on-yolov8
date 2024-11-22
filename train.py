# 导入'warnings'模块 忽略所有警告
import warnings

warnings.filterwarnings('ignore')
# 从'ultralytics'包导入'YOLO'类
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('E:/xiaobai/li-main/ultralytics/cfg/models/v8/my_model.yaml')
    model.train(data='E:/xiaobai/li-main/ultralytics/cfg/datasets/my_data.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=2,    # 32 / 16
                close_mosaic=0,
                workers=4,
                device='cpu',   # 0
                optimizer='SGD',
                project='runs/train',
                name='exp',
                )
