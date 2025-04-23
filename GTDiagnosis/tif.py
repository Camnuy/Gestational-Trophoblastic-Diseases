import sys
import json
import os
import torch
import torch.nn as nn
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QTextEdit, 
                             QFileDialog, QMessageBox, QSlider, QPushButton, QLineEdit, QFormLayout, QToolBar, QAction, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSplitter, QMenu, QComboBox, QDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QIcon
from PyQt5.QtCore import Qt, QRectF, QTimer
import cv2
import math
from datetime import datetime
from torchvision import transforms
from models.villi import VilliUNet
from PIL import Image, ImageFile, ImageDraw
from models.villi import HmNet
#import QTimer
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images

def smooth_edges(pre_label):
    blurred = cv2.GaussianBlur(pre_label, (31, 31), 5)
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    return binary

def remove_small_holes(pre_label, min_size=3000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pre_label, connectivity=8)
    sizes = stats[1:, -1]
    num_labels -= 1
    img_cleaned = np.zeros_like(pre_label)
    for i in range(0, num_labels):
        if sizes[i] >= min_size:
            img_cleaned[labels == i + 1] = 255
    return img_cleaned

def crop_image(model, image_path, device, patch_size=4096, step_size=None):
    Image.MAX_IMAGE_PIXELS = 4294836225  # 防止出现图片太大无法读取
    image = Image.open(image_path)
    width, height = image.size
    pre_label = np.zeros((height, width), dtype=np.uint8)
    if step_size is None:
        step_size = patch_size
    width_num = math.ceil(width / step_size)
    height_num = math.ceil(height / step_size)

    for i in range(0, width_num):
        for j in range(0, height_num):
            width_start = min(step_size * i, width - patch_size)
            height_start = min(step_size * j, height - patch_size)
            width_end = width_start + patch_size
            height_end = height_start + patch_size
            crop = image.crop((width_start, height_start, width_end, height_end))  # 修正此行
            crop = crop.resize((512, 512), Image.BILINEAR)
            pre_crop = predict(model=model, img=crop, device=device)
            pre_crop_img = Image.fromarray(pre_crop) 
            pre_crop_img = pre_crop_img.resize((patch_size, patch_size), Image.NEAREST)
            pre_crop = np.array(pre_crop_img) 
            pre_label[height_start:height_end, width_start:width_end] = pre_crop

    pre_label = Image.fromarray(pre_label)
    pre_label = pre_label.resize((width // 10, height // 10), Image.BILINEAR)
    pre_label = np.array(pre_label)
    pre_label = remove_small_holes(pre_label)
    pre_label = smooth_edges(pre_label)
    pre_label_img = Image.fromarray(pre_label)
    return pre_label_img



def crop_image_shuizhong(model, image_path, device, patch_size=4096, step_size=None):
    Image.MAX_IMAGE_PIXELS = 4294836225  # 防止出现图片太大无法读取
    image = Image.open(image_path)
    width, height = image.size
    pre_label = np.zeros((height, width), dtype=np.uint8)
    if step_size is None:
        step_size = patch_size
    width_num = math.ceil(width / step_size)
    height_num = math.ceil(height / step_size)

    for i in range(0, width_num):
        for j in range(0, height_num):
            width_start = min(step_size * i, width - patch_size)
            height_start = min(step_size * j, height - patch_size)
            width_end = width_start + patch_size
            height_end = height_start + patch_size
            crop = image.crop((width_start, height_start, width_end, height_end))
            crop = crop.resize((512, 512), Image.BILINEAR)
            pre_crop = predict(model=model, img=crop, device=device,index=0)
            pre_crop_img = Image.fromarray(pre_crop) 
            pre_crop_img = pre_crop_img.resize((patch_size, patch_size), Image.NEAREST)
            pre_crop = np.array(pre_crop_img) 
            pre_label[height_start:height_end, width_start:width_end] = pre_crop

    pre_label = Image.fromarray(pre_label)
    pre_label = pre_label.resize((width // 10, height // 10), Image.BILINEAR)
    pre_label = np.array(pre_label)
    pre_label = remove_small_holes(pre_label)
    pre_label = smooth_edges(pre_label)
    pre_label_img = Image.fromarray(pre_label)
    return pre_label_img

def crop_image_zengshengshuizhong(model, image_path, device, patch_size=4096, step_size=None):
    Image.MAX_IMAGE_PIXELS = 4294836225  # 防止出现图片太大无法读取
    image = Image.open(image_path)
    width, height = image.size
    pre_label_1 = np.zeros((height, width), dtype=np.uint8)
    pre_label_2 = np.zeros((height, width), dtype=np.uint8)
    if step_size is None:
        step_size = patch_size
    width_num = math.ceil(width / step_size)
    height_num = math.ceil(height / step_size)

    for i in range(0, width_num):
        for j in range(0, height_num):
            width_start = min(step_size * i, width - patch_size)
            height_start = min(step_size * j, height - patch_size)
            width_end = width_start + patch_size
            height_end = height_start + patch_size
            crop = image.crop((width_start, height_start, width_end, height_end))
            crop = crop.resize((512, 512), Image.BILINEAR)
            pre_crop_1,pre_crop_2 = predict_2(model=model, img=crop, device=device)
            pre_crop_img_1 = Image.fromarray(pre_crop_1) 
            pre_crop_img_1 = pre_crop_img_1.resize((patch_size, patch_size), Image.NEAREST)
            pre_crop_1 = np.array(pre_crop_img_1) 
            pre_label_1[height_start:height_end, width_start:width_end] = pre_crop_1
            pre_crop_img_2 = Image.fromarray(pre_crop_2) 
            pre_crop_img_2 = pre_crop_img_2.resize((patch_size, patch_size), Image.NEAREST)
            pre_crop_2 = np.array(pre_crop_img_2) 
            pre_label_2[height_start:height_end, width_start:width_end] = pre_crop_2

    pre_label_1 = Image.fromarray(pre_label_1)
    pre_label_1 = pre_label_1.resize((width // 10, height // 10), Image.BILINEAR)
    pre_label_1 = np.array(pre_label_1)
    pre_label_1 = remove_small_holes(pre_label_1)
    pre_label_1 = smooth_edges(pre_label_1)
    pre_label_img_1 = Image.fromarray(pre_label_1)
    pre_label_2 = Image.fromarray(pre_label_2)
    pre_label_2 = pre_label_2.resize((width // 10, height // 10), Image.BILINEAR)
    pre_label_2 = np.array(pre_label_2)
    pre_label_2 = remove_small_holes(pre_label_2)
    pre_label_2 = smooth_edges(pre_label_2)
    pre_label_img_2 = Image.fromarray(pre_label_2)

    return pre_label_img_1,pre_label_img_2

def predict_2(model, img, device):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        img = transform(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).cpu().squeeze().numpy()
        pred_1 = np.where(pred == 1, 255, 0).astype(np.uint8)
        pred_2 = np.where(pred == 2, 255, 0).astype(np.uint8)
        return pred_1,pred_2

def predict(model, img, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).cpu().squeeze().numpy()
    pred = np.where(pred == 1, 255, 0).astype(np.uint8)
    return pred
    
    

class PathologyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = VilliUNet()  # Initialize your model
        self.model.load_state_dict(torch.load("models/VilliUNet.pth", map_location="cpu"))  # Load your model weights
        self.model.eval()
        self.Hmodel = HmNet()
        self.Hmodel.load_state_dict(torch.load("models/HmNet.pth", map_location="cpu"))
        self.Hmodel.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.Hmodel = self.Hmodel.to(self.device)
        self.rongmao = None
        self.zengsheng = None
        self.shuizhong = None

        # Video processing related attributes
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_video_frame)
        self.stitched_image = None
        self.previous_image = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.orb = cv2.ORB_create()
        self.MAX_WIDTH = 5000
        self.MAX_HEIGHT = 5000
        self.transform_width = 0
        self.transform_height = 0

    def initUI(self):
        self.setWindowTitle('病理诊断软件')

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Create splitter for left, middle, and right layout
        horizontal_splitter = QSplitter(Qt.Horizontal)

        # Left side layout for logo and patient list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Logo label
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("logo.jpg")  # Replace with your logo path
        self.logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio))
        left_layout.addWidget(self.logo_label)
        
        # Patient list
        self.patient_list = QListWidget()
        self.patient_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.patient_list.customContextMenuRequested.connect(self.show_context_menu)
        self.patient_list.itemSelectionChanged.connect(self.load_patient_data)
        left_layout.addWidget(self.patient_list)
        
        left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(150)  # Set minimum width for the left panel
        horizontal_splitter.addWidget(left_widget)

        # Middle layout with vertical layout for two image display areas
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)

        horizontal_splitter_middle = QSplitter(Qt.Horizontal)

        # Left image display area
        left_image_widget = QWidget()
        left_image_layout = QVBoxLayout(left_image_widget)
        
        self.graphics_view_top = QGraphicsView()
        self.graphics_view_top.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_scene_top = QGraphicsScene()
        self.graphics_view_top.setScene(self.graphics_scene_top)
        left_image_layout.addWidget(self.graphics_view_top)
        
        self.zoom_slider_top = QSlider(Qt.Horizontal)
        self.zoom_slider_top.setMinimum(10)
        self.zoom_slider_top.setMaximum(200)
        self.zoom_slider_top.setValue(100)
        self.zoom_slider_top.valueChanged.connect(self.zoom_image_top)
        left_image_layout.addWidget(self.zoom_slider_top)
        
        left_image_widget.setLayout(left_image_layout)
        horizontal_splitter_middle.addWidget(left_image_widget)

        # Right image display area (for segmentation results)
        right_image_widget = QWidget()
        right_image_layout = QVBoxLayout(right_image_widget)
        
        self.graphics_view_bottom = QGraphicsView()
        self.graphics_view_bottom.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_scene_bottom = QGraphicsScene()
        self.graphics_view_bottom.setScene(self.graphics_scene_bottom)
        right_image_layout.addWidget(self.graphics_view_bottom)
        
        self.zoom_slider_bottom = QSlider(Qt.Horizontal)
        self.zoom_slider_bottom.setMinimum(10)
        self.zoom_slider_bottom.setMaximum(200)
        self.zoom_slider_bottom.setValue(100)
        self.zoom_slider_bottom.valueChanged.connect(self.zoom_image_bottom)
        right_image_layout.addWidget(self.zoom_slider_bottom)
        
        right_image_widget.setLayout(right_image_layout)
        horizontal_splitter_middle.addWidget(right_image_widget)

        middle_layout.addWidget(horizontal_splitter_middle)
        
        middle_widget.setLayout(middle_layout)
        horizontal_splitter.addWidget(middle_widget)

        # Right side layout for info and report
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Patient info
        self.info_form_layout = QFormLayout()
        self.info_form_layout.addRow(QLabel("患者基本信息"))

        self.name_edit = QLineEdit()
        self.gender_edit = QLineEdit()
        self.age_edit = QLineEdit()
        self.admission_time_edit = QLineEdit()
        self.bed_number_edit = QLineEdit()

        self.info_form_layout.addRow("姓名:", self.name_edit)
        self.info_form_layout.addRow("性别:", self.gender_edit)
        self.info_form_layout.addRow("年龄:", self.age_edit)
        self.info_form_layout.addRow("就诊时间:", self.admission_time_edit)
        self.info_form_layout.addRow("病房床号:", self.bed_number_edit)
        
        right_layout.addLayout(self.info_form_layout)

        self.update_info_button = QPushButton("更新基本信息")
        self.update_info_button.clicked.connect(self.update_patient_info)
        right_layout.addWidget(self.update_info_button)
        
        # Patient detailed info
        self.detailed_info_text = QTextEdit()
        right_layout.addWidget(self.detailed_info_text)
        
        # Diagnosis report
        self.report_text = QTextEdit()
        right_layout.addWidget(self.report_text)

        self.update_report_button = QPushButton("更新诊断报告")
        self.update_report_button.clicked.connect(self.update_diagnosis_report)
        right_layout.addWidget(self.update_report_button)

        right_widget.setLayout(right_layout)
        right_widget.setMinimumWidth(150)  # Set minimum width for the right panel
        horizontal_splitter.addWidget(right_widget)

        horizontal_splitter.setStretchFactor(0, 1)  # Set stretch factor to make sure the left panel is visible
        horizontal_splitter.setStretchFactor(1, 8)  # Set stretch factor for the middle panel to take more space
        horizontal_splitter.setStretchFactor(2, 1)  # Set stretch factor to make sure the right panel is visible
        
        main_layout.addWidget(horizontal_splitter)
        self.setCentralWidget(main_widget)
        
        # Create toolbar
        self.toolbar = QToolBar("主工具栏")
        self.addToolBar(self.toolbar)

        # Add actions to the toolbar
        add_patient_action = QAction(QIcon(), "添加新患者", self)
        add_patient_action.triggered.connect(self.add_new_patient)
        self.toolbar.addAction(add_patient_action)

        import_image_action = QAction(QIcon(), "导入新切片", self)
        import_image_action.triggered.connect(self.import_new_image)
        self.toolbar.addAction(import_image_action)

        # Previous and Next buttons
        self.prev_button = QAction("上一张", self)
        self.prev_button.triggered.connect(self.show_previous_image)
        self.toolbar.addAction(self.prev_button)

        self.next_button = QAction("下一张", self)
        self.next_button.triggered.connect(self.show_next_image)
        self.toolbar.addAction(self.next_button)

        # Run Segmentation with dropdown menu
        self.run_segmentation_action = QComboBox()
        self.run_segmentation_action.addItem("运行分割")
        self.run_segmentation_action.addItem("绒毛")
        self.run_segmentation_action.addItem("水肿")
        self.run_segmentation_action.addItem("增生")
        self.run_segmentation_action.currentIndexChanged.connect(self.run_segmentation)
        self.toolbar.addWidget(self.run_segmentation_action)

        self.diagnosis_action = QPushButton("诊断")
        self.diagnosis_action.clicked.connect(self.diagnosis)
        self.toolbar.addWidget(self.diagnosis_action)

        
        # Mode selection with dropdown menu
        self.mode_selection_action = QComboBox()
        self.mode_selection_action.addItem("读取切片模式")
        self.mode_selection_action.addItem("读取视频模式")
        self.mode_selection_action.currentIndexChanged.connect(self.change_mode)
        self.toolbar.addWidget(self.mode_selection_action)
        
        # Video controls
        self.select_video_button = QPushButton("选择视频文件")
        self.select_video_button.clicked.connect(self.select_video)
        self.toolbar.addWidget(self.select_video_button)
        
        self.play_pause_button = QPushButton("开始")
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.toggled.connect(self.play_pause_video)
        self.toolbar.addWidget(self.play_pause_button)
        
        # Load data
        self.data_folder = "data"
        self.image_item_top = None  # Initialize the image item
        self.image_item_bottom = None  # Initialize the image item
        self.current_image_index = 0  # Track the current image index
        self.current_image_path = None  # Track the current image path
        self.load_data()
    
    def calculate_metrics(self):
        total_pixels = self.rongmao.size[0]*self.rongmao.size[1]
        villi_pixels = np.sum(np.array(self.rongmao) == 255)
        edema_pixels = np.sum(np.array(self.shuizhong) == 255)
        hyperplasia_pixels = np.sum(np.array(self.zengsheng) == 255)
        print(villi_pixels)
        #假设绒毛1水肿2增生3
        #下面分别是绒毛比例 水肿比例 增生比例 异常绒毛比例
        villi_ratio = villi_pixels / total_pixels
        edema_ratio = edema_pixels / villi_pixels if villi_pixels > 0 else 0
        hyperplasia_ratio = hyperplasia_pixels / villi_pixels if villi_pixels > 0 else 0
        abnormal_villi_ratio = (edema_pixels + hyperplasia_pixels) / villi_pixels if villi_pixels > 0 else 0
        return {
            "total_pixels": total_pixels,
            "villi_pixels": villi_pixels,
            "edema_pixels": edema_pixels,
            "hyperplasia_pixels": hyperplasia_pixels,
            "villi_ratio": villi_ratio,
            "edema_ratio": edema_ratio,
            "hyperplasia_ratio": hyperplasia_ratio,
            "abnormal_villi_ratio": abnormal_villi_ratio
        }
    
    def classify_slide(self,metrics, villi_threshold1=0.001,villi_threshold2=0.05,edema_threshold1=0.1,edema_threshold2=0.3, hyperplasia_threshold1=0.1,hyperplasia_threshold2=0.3, abnormal_threshold=0.2):
        #基本没有绒毛
        print(metrics["villi_ratio"])
        if metrics["villi_ratio"] < villi_threshold1:
            return "视野内未见明显绒毛组织，请根据其他切片进行诊断"
        #很少量绒毛
        if villi_threshold1 <= metrics["villi_ratio"] < villi_threshold2:
            if metrics["edema_ratio"] > edema_threshold1 and metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
                return "视野内见少量胎盘绒毛及蜕膜组织，绒毛存在水肿及部分增生，建议根据其他切片图像进行进一步分析"
            elif metrics["edema_ratio"] >= edema_threshold1:
                return"视野内见少量胎盘绒毛及蜕膜组织，绒毛存在部分水肿，建议根据其他切片图像进行进一步分析"
            elif metrics["hyperplasia_ratio"] >= hyperplasia_threshold1:
                return "视野内见少量胎盘绒毛及蜕膜组织，绒毛存在部分增生，建议根据其他切片图像进行进一步分析"
            return "视野内见少量胎盘绒毛及蜕膜组织，无明显病灶，建议根据其他切片图像进行进一步分析"
        #有一些绒毛
        if metrics["villi_ratio"] >= villi_threshold2:
            if metrics["edema_ratio"] > edema_threshold2 and metrics["hyperplasia_ratio"] > hyperplasia_threshold2:
                return "视野内见胎盘绒毛及蜕膜组织，部分绒毛水肿明显，滋养层细胞可见增生，考虑葡萄胎，建议行分子STR检测进一步明确诊断。"
            elif metrics["edema_ratio"] > edema_threshold2 and metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
                return "视野内见胎盘绒毛及蜕膜组织，部分绒毛水肿明显，滋养层细胞局部增生，考虑葡萄胎，建议行分子STR检测进一步明确诊断。"
            elif metrics["edema_ratio"] > edema_threshold1 and metrics["hyperplasia_ratio"] > hyperplasia_threshold2:
                return "视野内见胎盘绒毛及蜕膜组织，部分绒毛水肿，滋养层细胞局部增生，考虑葡萄胎，建议行分子STR检测进一步明确诊断。"
            
            elif metrics["edema_ratio"] > edema_threshold1 and metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
                return"视野内见胎盘绒毛及蜕膜组织，绒毛存在轻度水肿，局灶滋养细胞增生，考虑为整体轻度异常，建议监测血HCG。"
            elif metrics["edema_ratio"] > edema_threshold1:
                return"视野内见胎盘绒毛及蜕膜组织，绒毛存在轻度水肿，考虑为整体轻度异常，建议监测血HCG。"
            elif metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
                return"视野内见胎盘绒毛及蜕膜组织，局灶滋养细胞增生，考虑为整体轻度异常，建议监测血HCG。"
            
            return "胎盘绒毛及蜕膜组织及部分子宫内膜，绒毛形态基本正常，考虑为正常流产"


    def generate_report(self,metrics, classification):
        report = f"诊断报告:\n\n"
        report += f"切片总面积: {metrics['total_pixels']}\n"
        report += f"绒毛组织面积: {metrics['villi_pixels']} ({metrics['villi_ratio']*100:.2f}%)\n"
        report += f"水肿组织面积: {metrics['edema_pixels']} ({metrics['edema_ratio']*100:.2f}%)\n"
        report += f"增生组织面积: {metrics['hyperplasia_pixels']} ({metrics['hyperplasia_ratio']*100:.2f}%)\n"
        report += f"异常绒毛占比: {metrics['abnormal_villi_ratio']*100:.2f}%\n\n"
        report += f"诊断结果: {classification}\n"
        return report
    def diagnosis(self):
        image_path = self.current_image_path
        if os.path.isfile(self.rongmao_path):
            self.rongmao = Image.open(self.rongmao_path)
        else:
            self.rongmao = crop_image(self.model, image_path, self.device).convert("L")
            self.rongmao.save(self.rongmao_path)
        if os.path.isfile(self.shuizhong_path) and os.path.isfile(self.zengsheng_path):
            self.shuizhong = Image.open(self.shuizhong_path)
            self.zengsheng = Image.open(self.zengsheng_path)
        else:
            self.shuizhong,self.zengsheng = crop_image_zengshengshuizhong(self.Hmodel, image_path, self.device)
            self.shuizhong = self.shuizhong.convert("L")
            self.zengsheng = self.zengsheng.convert("L")
            self.shuizhong.save(self.shuizhong_path)
            self.zengsheng.save(self.zengsheng_path)
        
        metrics = self.calculate_metrics()
        classification = self.classify_slide(metrics)
        report = self.generate_report(metrics, classification)

        self.detailed_info_text.setText(report)

    def load_data(self):
        if not os.path.exists(self.data_folder):
            QMessageBox.critical(self, '错误', f'数据文件夹 "{self.data_folder}" 不存在。')
            sys.exit()
        
        self.patients = []
        for patient_folder in os.listdir(self.data_folder):
            patient_path = os.path.join(self.data_folder, patient_folder)
            if os.path.isdir(patient_path):
                info_path = os.path.join(patient_path, 'info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r', encoding='utf-8') as f:
                        patient_data = json.load(f)
                        patient_data['folder'] = patient_folder  # Store folder name for image path
                        self.patients.append(patient_data)
                        self.patient_list.addItem(patient_data['name'])
                else:
                    print(f"未找到信息文件 {patient_folder}")
    
    def load_patient_data(self):
        self.mode_selection_action.setCurrentIndex(0)
        selected_items = self.patient_list.selectedItems()
        if selected_items:
            index = self.patient_list.row(selected_items[0])
            self.current_patient_data = self.patients[index]
            
            self.name_edit.setText(self.current_patient_data['name'])
            self.gender_edit.setText(self.current_patient_data['gender'])
            self.age_edit.setText(str(self.current_patient_data['age']))
            self.admission_time_edit.setText(self.current_patient_data['admission_time'])
            self.bed_number_edit.setText(self.current_patient_data['patient_id'])
            
            self.detailed_info_text.setText(
                f"本切片共包含绒毛{self.current_patient_data['villi_count']}个，"
                f"异常绒毛{self.current_patient_data['abnormal_villi_count']}个，"
                f"其中存在水肿病灶{self.current_patient_data['edema_foci_count']}处，"
                f"增生病灶{self.current_patient_data['hyperplasia_foci_count']}处，"
                f"异常绒毛占比{self.current_patient_data['abnormal_villi_ratio']}%。"
            )
            
            self.report_text.setText(self.current_patient_data['diagnosis_report'])
            
            # Load all images for the selected patient
            patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])
            self.image_files = [f for f in os.listdir(patient_folder) if f.endswith(('.tif', '.tiff')) and f != 'info.json']
            
            if self.image_files:
                self.current_image_index = 0
                self.display_image_top(self.image_files[0])
            else:
                self.graphics_scene_top.clear()
                QMessageBox.warning(self, '警告', f'未找到图像文件 {self.current_patient_data["name"]}')
    
    def display_image_top(self, image_file):
        self.run_segmentation_action.setCurrentIndex(0)
        patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])
        image_path = os.path.join(patient_folder, image_file)
        self.rongmao_path = image_path[:-4]+"_rongmao.png"
        self.shuizhong_path =image_path[:-4]+"_shuizhong.png"
        self.zengsheng_path =image_path[:-4]+"_zengsheng.png"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image = image.resize((image.width // 10, image.height // 10), Image.BILINEAR)
            image.save('resized_image.png')
            self.image_top = QPixmap('resized_image.png')
            self.image_item_top = QGraphicsPixmapItem(self.image_top)
            self.graphics_scene_top.clear()
            self.graphics_scene_top.addItem(self.image_item_top)
            self.graphics_scene_top.setSceneRect(QRectF(self.image_top.rect()).adjusted(-500, -500, 500, 500))
            self.zoom_image_top()  # Ensure to apply zoom on the new image
            
            # Update current image path
            self.current_image_path = image_path
            
            # Clear the bottom image display
            self.graphics_scene_bottom.clear()
            self.image_item_bottom = None
        

            

            

        else:
            self.graphics_scene_top.clear()
            self.image_item_top = None
            QMessageBox.warning(self, '警告', f'未找到图像文件: {image_file}')
        
    
    def zoom_image_top(self):
        if self.image_item_top:
            scale_factor = self.zoom_slider_top.value() / 100.0
            self.image_item_top.setScale(scale_factor)
    
    def zoom_image_bottom(self):
        if self.image_item_bottom:
            scale_factor = self.zoom_slider_bottom.value() / 100.0
            self.image_item_bottom.setScale(scale_factor)

    def update_patient_info(self):
        self.current_patient_data['name'] = self.name_edit.text()
        self.current_patient_data['gender'] = self.gender_edit.text()
        self.current_patient_data['age'] = int(self.age_edit.text())
        self.current_patient_data['admission_time'] = self.admission_time_edit.text()
        self.current_patient_data['patient_id'] = self.bed_number_edit.text()
        
        patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])
        info_path = os.path.join(patient_folder, 'info.json')
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_patient_data, f, ensure_ascii=False, indent=4)
        
        # Update the patient's name in the list
        current_row = self.patient_list.currentRow()
        self.patient_list.item(current_row).setText(self.current_patient_data['name'])

        QMessageBox.information(self, '信息', '患者基本信息已更新。')
    
    def update_diagnosis_report(self):
        self.current_patient_data['diagnosis_report'] = self.report_text.toPlainText()
        
        patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])
        info_path = os.path.join(patient_folder, 'info.json')
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_patient_data, f, ensure_ascii=False, indent=4)
        
        QMessageBox.information(self, '信息', '诊断报告已更新。')
    
    def add_new_patient(self):
        new_patient_name = self.generate_patient_id()
        new_patient_folder = os.path.join(self.data_folder, new_patient_name)
        if not os.path.exists(new_patient_folder):
            os.makedirs(new_patient_folder)
        new_patient_info = {
            'name': new_patient_name,
            'gender': '',
            'age': 0,
            'admission_time': '',
            'patient_id': '',
            'villi_count': 0,
            'abnormal_villi_count': 0,
            'edema_foci_count': 0,
            'hyperplasia_foci_count': 0,
            'abnormal_villi_ratio': 0.0,
            'diagnosis_report': '',
            'folder': new_patient_name
        }
        info_path = os.path.join(new_patient_folder, 'info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(new_patient_info, f, ensure_ascii=False, indent=4)
        self.patients.append(new_patient_info)
        self.patient_list.addItem(new_patient_name)
        self.patient_list.setCurrentRow(len(self.patients) - 1)
        self.clear_patient_info()
        QMessageBox.information(self, '信息', f'新患者 {new_patient_name} 已添加。')

    def generate_patient_id(self):
        existing_ids = [int(folder) for folder in os.listdir(self.data_folder) if folder.isdigit()]
        if existing_ids:
            new_id = max(existing_ids) + 1
        else:
            new_id = 1
        return f"{new_id:04d}"
    
    def clear_patient_info(self):
        self.name_edit.clear()
        self.gender_edit.clear()
        self.age_edit.clear()
        self.admission_time_edit.clear()
        self.bed_number_edit.clear()
        self.detailed_info_text.clear()
        self.report_text.clear()
        self.graphics_scene_top.clear()
        self.graphics_scene_bottom.clear()
        self.image_item_top = None
        self.image_item_bottom = None
        self.zoom_slider_top.setValue(100)
        self.zoom_slider_bottom.setValue(100)

    def import_new_image(self):
        selected_items = self.patient_list.selectedItems()
        if selected_items:
            index = self.patient_list.row(selected_items[0])
            patient_folder = os.path.join(self.data_folder, self.patients[index]['folder'])
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "导入新切片", "", "Images (*.tif *.tiff)", options=options)
            if file_name:
                _, file_ext = os.path.splitext(file_name)
                new_image_name = f'image{len(self.image_files) + 1}{file_ext}'
                new_image_path = os.path.join(patient_folder, new_image_name)
                os.rename(file_name, new_image_path)
                self.image_files.append(new_image_name)
                self.display_image_top(new_image_name)
                QMessageBox.information(self, '信息', f'新切片已导入 {new_image_name}。')

    def show_previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image_top(self.image_files[self.current_image_index])

    def show_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_image_top(self.image_files[self.current_image_index])

    def show_context_menu(self, pos):
        context_menu = QMenu(self)
        delete_action = QAction("删除", self)
        delete_action.triggered.connect(self.delete_patient)
        context_menu.addAction(delete_action)
        context_menu.exec_(self.patient_list.mapToGlobal(pos))

    def delete_patient(self):
        selected_items = self.patient_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            index = self.patient_list.row(item)
            patient_data = self.patients[index]
            patient_folder = os.path.join(self.data_folder, patient_data['folder'])

            # Confirm deletion
            reply = QMessageBox.question(self, '确认删除', f'确定要删除患者 {patient_data["name"]} 的全部信息吗？', 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Remove the patient folder
                if os.path.exists(patient_folder):
                    for root, dirs, files in os.walk(patient_folder, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(patient_folder)

                # Remove from the list
                self.patient_list.takeItem(index)
                del self.patients[index]

                QMessageBox.information(self, '信息', f'患者 {patient_data["name"]} 的全部信息已删除。')

    def run_segmentation(self, index):
        if(index == 0):
            return
        selected_items = self.patient_list.selectedItems()
        if selected_items and self.current_image_path:
            image_path = self.current_image_path

            if os.path.exists(image_path):
                # 打开原图像
                original_image = Image.open(image_path).convert("RGBA")
                original_width, original_height = original_image.size
                new_width = int(original_width * 0.1)
                new_height = int(original_height * 0.1)
                original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # 确保尺寸一致
                
                # 获取分割结果图像
                if(index == 1):
                    if os.path.isfile(self.rongmao_path):
                        self.rongmao = Image.open(self.rongmao_path)
                    else:
                        self.rongmao = crop_image(self.model, image_path, self.device).convert("L")
                        self.rongmao.save(self.rongmao_path)
                    segmentation_result = self.rongmao
                if(index == 2):
                    if os.path.isfile(self.shuizhong_path) and os.path.isfile(self.zengsheng_path):
                        self.shuizhong = Image.open(self.shuizhong_path)
                        self.zengsheng = Image.open(self.zengsheng_path)
                    else:
                        self.shuizhong,self.zengsheng = crop_image_zengshengshuizhong(self.Hmodel, image_path, self.device)
                        self.shuizhong = self.shuizhong.convert("L")
                        self.zengsheng = self.zengsheng.convert("L")
                        self.shuizhong.save(self.shuizhong_path)
                        self.zengsheng.save(self.zengsheng_path)
                    segmentation_result = self.shuizhong
                if(index == 3):
                    if os.path.isfile(self.shuizhong_path) and os.path.isfile(self.zengsheng_path):
                        self.shuizhong = Image.open(self.shuizhong_path)
                        self.zengsheng = Image.open(self.zengsheng_path)
                    else:
                        self.shuizhong,self.zengsheng = crop_image_zengshengshuizhong(self.Hmodel, image_path, self.device)
                        self.shuizhong = self.shuizhong.convert("L")
                        self.zengsheng = self.zengsheng.convert("L")
                        self.shuizhong.save(self.shuizhong_path)
                        self.zengsheng.save(self.zengsheng_path)
                    segmentation_result = self.zengsheng
                segmentation_result = segmentation_result.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 创建红色透明叠加层
                overlay = Image.new("RGBA", original_image.size, (255, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                for y in range(segmentation_result.height):
                    for x in range(segmentation_result.width):
                        if segmentation_result.getpixel((x, y)) == 255:  # 分类为1的部分
                            overlay.putpixel((x, y), (255, 0, 0, 128))  # 透明红色

                # 将叠加层与原图像合并
                combined_image = Image.alpha_composite(original_image, overlay)
                
                # 保存和显示结果图像
                combined_image.save('combined_image.png')
                self.image_bottom = QPixmap('combined_image.png')
                self.image_item_bottom = QGraphicsPixmapItem(self.image_bottom)
                self.graphics_scene_bottom.clear()
                self.graphics_scene_bottom.addItem(self.image_item_bottom)
                self.graphics_scene_bottom.setSceneRect(QRectF(self.image_bottom.rect()).adjusted(-500, -500, 500, 500))
                self.zoom_image_bottom()

    
    def select_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Videos (*.mp4 *.avi *.mkv)", options=options)
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.stitched_image = None
            self.previous_image = None
            self.previous_keypoints = None
            self.previous_descriptors = None
            self.transform_width = 0
            self.transform_height = 0

    def play_pause_video(self, checked):
        if checked:
            self.play_pause_button.setText("暂停")
            self.timer.start(30)  # 每30毫秒处理一帧
        else:
            self.play_pause_button.setText("开始")
            self.timer.stop()

    def process_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if self.previous_image is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.previous_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([self.previous_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            self.transform_width += H[0, 2]
            self.transform_height += H[1, 2]
            
            if self.stitched_image is None:
                self.stitched_image = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH, 3), dtype=np.uint8)
                center_x = (self.MAX_WIDTH - self.previous_image.shape[1]) // 2
                center_y = (self.MAX_HEIGHT - self.previous_image.shape[0]) // 2
                self.stitched_image[center_y:center_y + self.previous_image.shape[0], center_x:center_x + self.previous_image.shape[1]] = self.previous_frame
            
            center_x = (self.MAX_WIDTH - frame.shape[1]) // 2
            center_y = (self.MAX_HEIGHT - frame.shape[0]) // 2
            start_x = int(center_x - self.transform_width)
            start_y = int(center_y - self.transform_height)
            end_x = start_x + frame.shape[1]
            end_y = start_y + frame.shape[0]

            if start_x < 0 or start_y < 0 or end_x > self.MAX_WIDTH or end_y > self.MAX_HEIGHT:
                print("Transformations exceed bounds of stitched image.")
                self.timer.stop()
                return
            
            self.stitched_image[start_y:end_y, start_x:end_x] = frame
            
            segmentation_output = self.segment_frame(frame)
            self.stitched_image[start_y:end_y, start_x:end_x] = self.apply_segmentation_overlay(self.stitched_image[start_y:end_y, start_x:end_x], segmentation_output, alpha=0.5)

        segmentation_output = self.segment_frame(frame)
        frame = self.apply_segmentation_overlay(frame, segmentation_output, alpha=0.5)
        
        self.previous_frame = frame
        self.previous_image = gray
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

        self.update_frame_display(self.graphics_scene_top, frame)
        if self.stitched_image is not None:
            self.update_frame_display(self.graphics_scene_bottom, self.stitched_image)

    def update_frame_display(self, scene, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scene.clear()
        scene.addPixmap(pixmap)
        scene.setSceneRect(QRectF(pixmap.rect()))  # 设置场景矩形以使拖动有效

    def segment_frame(self, frame):
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = cv2.resize(input_frame, (256, 256))  # 根据模型输入尺寸调整
        input_tensor = torch.from_numpy(input_frame).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

        with torch.no_grad():
            output = self.model(input_tensor)
        
        output = output.cpu().numpy()[0, 1, :, :]  # 获取类别为1的概率
        output = cv2.resize(output, (frame.shape[1], frame.shape[0]))  # 调整回原始尺寸
        return output

    def apply_segmentation_overlay(self, image, segmentation_output, alpha=0.5):
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[segmentation_output > 0.5] = [0, 0, 255]  # 红色标记
        mask = segmentation_output > 0.5
        image[mask] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask]
        return image

    def change_mode(self, index):
        if index == 1:  # Video mode
            self.graphics_scene_top.clear()
            self.graphics_scene_bottom.clear()
            self.image_item_top = None
            self.image_item_bottom = None
            self.zoom_slider_top.setValue(100)
            self.zoom_slider_bottom.setValue(100)

    def zoom_image_top(self):
        if self.image_item_top:
            scale_factor = self.zoom_slider_top.value() / 100.0
            self.graphics_view_top.resetTransform()
            self.graphics_view_top.scale(scale_factor, scale_factor)
        else:
            self.graphics_view_top.resetTransform()
            scale_factor = self.zoom_slider_top.value() / 100.0
            self.graphics_view_top.scale(scale_factor, scale_factor)

    def zoom_image_bottom(self):
        if self.image_item_bottom:
            scale_factor = self.zoom_slider_bottom.value() / 100.0
            self.graphics_view_bottom.resetTransform()
            self.graphics_view_bottom.scale(scale_factor, scale_factor)
        else:
            self.graphics_view_bottom.resetTransform()
            scale_factor = self.zoom_slider_bottom.value() / 100.0
            self.graphics_view_bottom.scale(scale_factor, scale_factor)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PathologyApp()
    ex.show()
    sys.exit(app.exec_())
