from ultralytics import YOLOv10
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载模型
model = YOLOv10("yolov10n-test.yaml")  # 模型结构
model = YOLOv10("yolov10n.pt")  # 加载预训练权重
if __name__ == '__main__':
    model.train(data="data.yaml", imgsz=640, batch=16, epochs=10, workers=0)  # 训练模型
