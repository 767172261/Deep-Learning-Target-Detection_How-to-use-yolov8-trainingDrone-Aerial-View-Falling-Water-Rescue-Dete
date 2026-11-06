如何yolov8训练使用——无人机视角下落水救援检测数据集 并构建检测溺水落水检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52652092c56449b99db42ce0e70c84d6.png)
如何做一个——无人机视角下落水救援检测数据集，利用无人机快速搜索落水者对增加受害者的生存机会至关重要，该数据集共收集12万帧视频图像，涵盖无人机高度从10m-60m高度，检测包括落水者（11万标注量）、流木（9000标注量）、救生圈（10000标注量）、冲浪板（2000标注量）、小船（30000标注量）共5类目标，数据量244GB，支持json，voc，yolo格式文件![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0160390b64fc44a4a7d1c8b9da313d84.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6119f3eacf49405dbe19fa9407498ac7.png)
构建一个基于YOLOv8的无人机视角下落水救援检测系统。详细的步骤和代码示例，包括数据集准备、环境部署、模型训练、指标可视化展示以及PyQt5界面设计。

### 数据集中的标签及数量表格

| 目标类型   | 标注量 |
|------------|--------|
| 落水者     | 110,000|
| 流木       | 9,000  |
| 救生圈     | 10,000 |
| 冲浪板     | 2,000  |
| 小船       | 30,000 |

### 环境部署说明

首先，确保你已经安装了必要的库。以下是详细的环境部署步骤：

#### 安装依赖

```bash
# 创建虚拟环境（可选）
conda create -n drone_rescue_env python=3.9
conda activate drone_rescue_env

# 安装PyTorch
pip install torch==1.9 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# 安装其他依赖
pip install opencv-python pyqt5 ultralytics moviepy scikit-learn pandas matplotlib seaborn
```

### 数据集准备

假设你的数据集已经准备好，并且是以YOLO格式存储的。我们将主要使用YOLO格式进行训练。

#### 数据集结构
```
dataset/
├── images/
│   ├── train/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
│   └── val/
│       ├── frame3.jpg
│       ├── frame4.jpg
│       └── ...
├── labels/
│   ├── train/
│       ├── frame1.txt
│       ├── frame2.txt
│       └── ...
│   └── val/
│       ├── frame3.txt
│       ├── frame4.txt
│       └── ...
└── classes.txt
```

`classes.txt` 内容如下：
```
person_in_water
driftwood
lifebuoy
surfboard
small_boat
```

每个图像对应的标签文件是一个文本文件，每行表示一个边界框，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```

### 视频到帧的转换

我们需要将视频转换为帧，并提取标注信息。

#### 视频转帧脚本 `video_to_frames.py`

```python
[<title="Convert Videos to Frames and Extract Annotations">]
import os
import cv2
from moviepy.editor import VideoFileClip

def extract_frames(video_path, output_dir, annotations):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"frame{frame_count}.jpg"
        frame_filepath = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_filepath, frame)
        
        if frame_count in annotations:
            annotation_file = frame_filepath.replace('.jpg', '.txt')
            with open(annotation_file, 'w') as f:
                for bbox in annotations[frame_count]:
                    class_id, x_center, y_center, width, height = bbox
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        frame_count += 1
    
    cap.release()

def main():
    dataset_path = 'path/to/dataset'
    videos_path = os.path.join(dataset_path, 'videos')
    frames_output_path_train = os.path.join(dataset_path, 'images/train')
    frames_output_path_val = os.path.join(dataset_path, 'images/val')
    labels_output_path_train = os.path.join(dataset_path, 'labels/train')
    labels_output_path_val = os.path.join(dataset_path, 'labels/val')
    
    os.makedirs(frames_output_path_train, exist_ok=True)
    os.makedirs(frames_output_path_val, exist_ok=True)
    os.makedirs(labels_output_path_train, exist_ok=True)
    os.makedirs(labels_output_path_val, exist_ok=True)
    
    # Example annotations dictionary
    annotations = {
        'video1.mp4': {
            0: [(0, 0.5, 0.5, 0.2, 0.2)],
            1: [(1, 0.4, 0.4, 0.1, 0.1)]
        },
        'video2.mp4': {
            0: [(2, 0.3, 0.3, 0.15, 0.15)],
            1: [(3, 0.2, 0.2, 0.1, 0.1)]
        }
    }
    
    for video_name in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video_name)
        if video_name.endswith('.mp4'):
            if video_name.startswith('train_'):
                extract_frames(video_path, frames_output_path_train, annotations.get(video_name, {}))
            elif video_name.startswith('val_'):
                extract_frames(video_path, frames_output_path_val, annotations.get(video_name, {}))

if __name__ == "__main__":
    main()
```

请将 `path/to/dataset` 替换为实际的数据集路径，并根据实际情况调整 `annotations` 字典。

### 模型训练权重和指标可视化展示

我们将使用YOLOv8进行训练，并在训练过程中记录各种指标，如F1曲线、准确率、召回率、损失曲线和混淆矩阵。

#### 训练脚本 `train_yolov8.py`

```python
[<title="Training YOLOv8 on Drone Rescue Detection Dataset">]
from ultralytics import YOLO
import os

# Define paths
dataset_path = 'path/to/dataset'
weights_path = 'best.pt'

# Create dataset.yaml
yaml_content = f"""
train: {os.path.join(dataset_path, 'images/train')}
val: {os.path.join(dataset_path, 'images/val')}

nc: 5
names: ['person_in_water', 'driftwood', 'lifebuoy', 'surfboard', 'small_boat']
"""

with open(os.path.join(dataset_path, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

# Train YOLOv8
model = YOLO('yolov8n.pt')  # Load a pretrained model (recommended for training)
results = model.train(data=os.path.join(dataset_path, 'dataset.yaml'), epochs=100, imgsz=640, save=True)

# Save the best weights
model.export(format='pt')
os.rename('runs/detect/train/weights/best.pt', weights_path)
```

请将 `path/to/dataset` 替换为实际的数据集路径。

### 指标可视化展示

我们将编写代码来可视化训练过程中的各项指标，包括F1曲线、准确率、召回率、损失曲线和混淆矩阵。

#### 可视化脚本 `visualize_metrics.py`

```python
[<title="Visualizing Training Metrics for YOLOv8">]
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load metrics
metrics_path = 'runs/detect/train/metrics.json'
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract metrics
loss = [entry['loss'] for entry in metrics if 'loss' in entry]
precision = [entry['metrics/precision(m)'] for entry in metrics if 'metrics/precision(m)' in entry]
recall = [entry['metrics/recall(m)'] for entry in metrics if 'metrics/recall(m)' in entry]
f1 = [entry['metrics/mAP50(m)'] for entry in metrics if 'metrics/mAP50(m)' in entry]

# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# Plot precision and recall curves
plt.subplot(1, 3, 2)
plt.plot(precision, label='Precision')
plt.plot(recall, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Precision and Recall Curves')
plt.legend()

# Plot F1 curve
plt.subplot(1, 3, 3)
plt.plot(f1, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion matrix
# Assuming you have predictions and true labels
# For demonstration, let's create some dummy data
true_labels = np.random.randint(0, 6, size=100)  # 0 to 5 (background or one of the object types)
predictions = np.random.randint(0, 6, size=100)  # 0 to 5 (background or one of the object types)

cm = confusion_matrix(true_labels, predictions, labels=list(range(6)))
labels = ['Background', 'person_in_water', 'driftwood', 'lifebuoy', 'surfboard', 'small_boat']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```

### PyQt5设计的界面

我们将使用PyQt5设计一个简单的GUI界面来进行模型预测。

#### GUI代码 `gui_app.py`

```python
[<title="PyQt5 GUI for YOLOv8 Drone Rescue Detection">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QProgressBar, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from ui_mainwindow import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Drone Rescue Detection")
        self.setGeometry(100, 100, 800, 600)

        self.model = YOLO('models/best.pt')
        self.cap = None
        self.timer = QTimer()

        self.ui.pushButton_image.clicked.connect(self.open_image)
        self.ui.pushButton_folder.clicked.connect(self.open_folder)
        self.ui.pushButton_video.clicked.connect(self.open_video)
        self.ui.pushButton_camera.clicked.connect(self.start_camera)
        self.ui.pushButton_stop.clicked.connect(self.stop_camera)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.jpeg *.jpg);;All Files (*)", options=options)
        if file_name:
            self.process_image(file_name)

    def process_image(self, file_name):
        img = cv2.imread(file_name)  # BGR
        assert img is not None, f'Image Not Found {file_name}'

        results = self.model(img, stream=True)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = f'{self.model.names[cls]} {conf:.2f}'
                color = (0, 255, 0)  # Green
                cv2.rectangle(img, r[:2], r[2:], color, 2)
                cv2.putText(img, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.label_display.setPixmap(pixmap.scaled(800, 600))

    def open_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_name:
            for filename in os.listdir(folder_name):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_name, filename)
                    self.process_image(file_path)

    def open_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Videos (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(30)  # Process frame every 30 ms

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # Process frame every 30 ms

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.timer.stop()

    def process_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame, stream=True)

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        label = f'{self.model.names[cls]} {conf:.2f}'
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, r[:2], r[2:], color, 2)
                        cv2.putText(frame, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.ui.label_display.setPixmap(pixmap.scaled(800, 600))
            else:
                self.cap.release()
                self.cap = None
                self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

### 辅助工具文件 `utils.py`

这个文件可以用来存放一些辅助函数，比如保存结果等。

```python
[<title="Utility Functions for Drone Rescue Detection">]
import cv2
import os

def save_results(image, detections, output_dir, filename):
    for det in detections:
        r = det['bbox']
        cls = det['class']
        conf = det['confidence']
        label = f'{cls} {conf:.2f}'
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 2)
        cv2.putText(image, label, (int(r[0]), int(r[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
```

### 界面设计文件 `ui_mainwindow.py`

假设你已经使用 `pyuic5` 将 `ui_mainwindow.ui` 转换为 `ui_mainwindow.py`。这里提供一个简化的示例。

```python
[<title="Generated UI Code from ui_mainwindow.ui">]
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_display = QtWidgets.QLabel(self.centralwidget)
        self.label_display.setText("")
        self.label_display.setObjectName("label_display")
        self.verticalLayout.addWidget(self.label_display)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_image = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_image.setObjectName("pushButton_image")
        self.horizontalLayout.addWidget(self.pushButton_image)
        self.pushButton_folder = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_folder.setObjectName("pushButton_folder")
        self.horizontalLayout.addWidget(self.pushButton_folder)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.setObjectName("pushButton_video")
        self.horizontalLayout.addWidget(self.pushButton_video)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.horizontalLayout.addWidget(self.pushButton_camera)
        self.pushButton_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.horizontalLayout.addWidget(self.pushButton_stop)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_image.setText(_translate("MainWindow", "Open Image"))
        self.pushButton_folder.setText(_translate("MainWindow", "Open Folder"))
        self.pushButton_video.setText(_translate("MainWindow", "Open Video"))
        self.pushButton_camera.setText(_translate("MainWindow", "Start Camera"))
        self.pushButton_stop.setText(_translate("MainWindow", "Stop Camera"))
```

### 资源文件 `resources.qrc` 和 `resources_rc.py`

假设你已经创建了一个 `resources.qrc` 文件，并使用 `pyrcc5` 转换为 `resources_rc.py`。

```xml
<RCC>
    <qresource prefix="/">
        <file>icons/icon.png</file>
    </qresource>
</RCC>
```

### 运行效果展示

假设你已经有了运行效果的图像，可以在 `README.md` 中添加这些图像以供参考。

```markdown
# Drone Rescue Detection System

## Overview
This project provides a deep learning-based system for detecting various objects related to rescue operations using drones equipped with cameras. The system can identify targets such as people in water, driftwood, lifebuoys, surfboards, and small boats.

## Environment Setup
- Software: PyCharm + Anaconda
- Environment: Python=3.9, OpenCV-Python, PyQt5, Torch=1.9

## Features
- Detects 5 types of objects: ["person_in_water", 'driftwood', 'lifebuoy', 'surfboard', 'small_boat']
- Supports detection on images, folders, videos, and live camera feed.
- Batch processing of images.
- Real-time display of detected objects with confidence scores and bounding boxes.
- Saving detection results.

## Usage
1. Run the program.
2. Choose an option to detect objects in images, folders, videos, or via the camera.

## Screenshots
![Example Screenshot](data/screenshots/example_screenshot.png)
```

### 总结

构建一个完整的基于YOLOv8的无人机视角下落水救援检测系统，包括数据集准备、环境部署、模型训练、指标可视化展示和PyQt5界面设计。以下是所有相关的代码文件：

1. **视频转帧脚本** (`video_to_frames.py`)
2. **训练脚本** (`train_yolov8.py`)
3. **指标可视化脚本** (`visualize_metrics.py`)
4. **GUI应用代码** (`gui_app.py`)
5. **辅助工具文件** (`utils.py`)
6. **UI界面源文件** (`ui_mainwindow.py`)
7. **资源文件** (`resources.qrc`, `resources_rc.py`)
8. **文档** (`README.md`)


