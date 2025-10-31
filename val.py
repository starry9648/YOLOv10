from ultralytics import YOLOv10


def main():
    # 加载模型，split='test'利用测试集进行测试
    model = YOLOv10(r"runs/detect/train4/weights/best.pt")
    model.val(data="data.yaml", split='test', imgsz=640, batch=16, device=0, workers=8)  # 模型验证


if __name__ == "__main__":
    main()
