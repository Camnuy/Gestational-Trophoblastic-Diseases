import sys
import json
import os
import torch
from PyQt5.QtWidgets import (QMessageBox, QMainWindow )
from PyQt5.QtCore import Qt, QTimer
import cv2
from models.villi import VilliUNet
from PIL import Image, ImageFile
from PyQt5.QtCore import QFile, QTextStream
from PyQt5.QtGui import QPixmap, QPainter
#import QTimer
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images

from model_diagnosis import *
from model_display import *
from model_segmentation import *
from model_update import *
from model_video import *
from model_online import *
# from ui_initialization_ori import *
from ui_initialization import *

class PathologyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_styles()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VilliUNet()  # Initialize your model
        self.model.load_state_dict(torch.load("models/VilliUNet.pth", map_location=self.device))  # Load your model weights
        self.model.eval()
        self.zengmodel = torch.jit.load('models/zengsheng.pt', map_location=self.device)
        self.zengmodel.eval()
        self.shuimodel = torch.jit.load('models/shuizhong.pt', map_location=self.device)
        self.shuimodel.eval()
        
        self.model = self.model.to(self.device)
        self.zengmodel = self.zengmodel.to(self.device)
        self.shuimodel = self.shuimodel.to(self.device)
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
        self.mode = 1 #1为切片模式，2为显微镜模式

    def load_styles(self):
        # 加载样式表
        file = QFile("styles.qss")
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        self.setStyleSheet(stream.readAll())

    def initUI(self):
        initUI(self)
    
    def calculate_metrics(self):
        return calculate_metrics(self)
    
    def calculate_metrics_with_count(self):
        return calculate_metrics_with_count(self)
        
    def classify_slide(self,metrics, villi_threshold1=0.001,
                       villi_threshold2=0.05,edema_threshold1=0.1,
                       edema_threshold2=0.3, hyperplasia_threshold1=0.1,
                       hyperplasia_threshold2=0.3, abnormal_threshold=0.2):
        return classify_slide(self,metrics, villi_threshold1, villi_threshold2,edema_threshold1,
                       edema_threshold2, hyperplasia_threshold1, hyperplasia_threshold2, 
                       abnormal_threshold)

    def generate_report(self, metrics, classification):
        return generate_report(self, metrics, classification)
        
    
    def generate_pie_chart(self, metrics):
        return generate_pie_chart(self, metrics)
        
    def plot_pie_charts(self ,metrics):
        return plot_pie_charts(self ,metrics)
    
    def diagnosis(self):
        diagnosis(self)

    def load_data(self):
        load_data(self)

    def load_patient_data(self):
        load_patient_data(self)

    
    def on_item_expanded(self, item):
        on_item_expanded(self, item)
        
    def display_image_lesion(self):
        display_image_lesion(self)

    def display_image_top(self, image_file):
        display_image_top(self, image_file)
    
    # Function to take screenshot
    def take_screenshot(self):
        take_screenshot(self)
    
    def show_full_image(self, image_path):
        show_full_image(self, image_path)
           
    def zoom_image_top(self):
        zoom_image_top(self)
    
    def zoom_image_bottom(self):
        zoom_image_bottom(self)

    def update_patient_info(self):
        update_patient_info(self)
    
    def update_diagnosis_report(self):
        update_diagnosis_report(self)
    
    def generate_patient_id(self):
        generate_patient_id(self)

    def add_new_patient(self):
        add_new_patient(self)

    def add_new_patient_group(self):
        add_new_patient_group(self)
    
    def clear_patient_info(self):
        clear_patient_info(self)

    def import_new_image(self):
        import_new_image(self)
    
    def import_new_image_group(self):
        import_new_image_group(self)
        
    def show_previous_image(self):
        show_previous_image(self)

    def show_next_image(self):
        show_next_image(self)

    def show_context_menu(self, pos):
        show_context_menu(self, pos)

    def delete_patient(self):
        delete_patient(self)

    def run_segmentation(self, index):
        return run_segmentation(self, index)
    def process_video_frame(self):
        return process_video_frame(self)

    def update_frame_display(self, scene, image):
        update_frame_display(self, scene, image)

    def segment_frame(self, frame):
        return segment_frame(self, frame)
    
    def initialize_right_panel(self):
        return initialize_right_panel(self)

    def apply_segmentation_overlay(self, image, segmentation_output, alpha=0.5):
        return apply_segmentation_overlay(self, image, segmentation_output, alpha)
    
    def toggle_segmentation(self):
        toggle_segmentation(self)
    
    def clear_right_image(self):
        clear_right_image(self)

    def open_help_pdf(self):
        open_help_pdf(self)
    
    def open_dev_info_pdf(self):
        open_dev_info_pdf(self)

    def online_learn(self):
        online_learn(self)

    def change_mode(self, button):
        if button == self.radio_button_microscope:
            self.graphics_view_top.scene.clear()
            self.graphics_view_bottom.scene.clear()
            self.image_item_top = None
            self.image_item_bottom = None
            self.zoom_slider_top.setValue(100)
            self.zoom_slider_bottom.setValue(100)
            self.cap = cv2.VideoCapture(self.camera_index)
            self.stitched_image = None
            self.previous_image = None
            self.previous_keypoints = None
            self.previous_descriptors = None
            self.transform_width = 0
            self.transform_height = 0
            self.del_border = 10 
            self.timer.start(30)
            self.whether_label = 0
            self.toggle_segmentation_button.setChecked(False)

            # 显微镜模式：隐藏智能分割下拉框，显示开关键
            self.checkbox_widget.setVisible(False)
            self.toggle_and_clear_widget.setVisible(True)
            # 隐藏两个切片添加按钮
            self.import_image_button.setVisible(False)
            self.import_image_group_button.setVisible(False)
            self.mode = 2

        elif button == self.radio_button_slices:
            self.graphics_view_top.scene.clear()
            self.graphics_view_bottom.scene.clear()
            self.image_item_top = None
            self.image_item_bottom = None
            self.timer.stop()

            # 切片扫描模式：显示智能分割下拉框，隐藏开关键，隐藏清空按钮
            self.checkbox_widget.setVisible(True)
            self.toggle_and_clear_widget.setVisible(False)
            # 显示两个切片添加按钮
            self.import_image_button.setVisible(True)
            self.import_image_group_button.setVisible(True)
            self.mode = 1

    
    def update_slider_value(self, view):
        update_slider_value(self, view)

        
    def showFullScreen(self):
        self.setWindowState(Qt.WindowFullScreen)

    def run_segmentation_online(self,index):
        run_segmentation_online(self,index)
    def display_image_bottom(self, image_path):
        display_image_bottom(self, image_path)

    def add_overlay_bottom(self, image_path, overlay_path, overlay_type):
        add_overlay_bottom(self, image_path, overlay_path, overlay_type)

    def upload_image(self,file_path, index):
        upload_image(self,file_path, index)
    
    def select_camera_index(self,index):
        self.camera_index = select_camera(self,index)
    
    def show_added_to_database_dialog(self):
        show_added_to_database_dialog(self)
    
    def add_overlay(self, scene, overlay_path, color):
        add_overlay(self, scene, overlay_path, color)

    def change_color(self, button):
        change_color(self, button)


    def save_and_crop_images(self, stitched_image, zengsheng, shuizhong, rongmao):
        save_and_crop_images(self, stitched_image, zengsheng, shuizhong, rongmao)

    def crop_to_content_with_coords(self, img):
        crop_to_content_with_coords(self, img)

    def change_language(self):
        change_language(self)

    def translate_to_english(self):
        translate_to_english(self)
        
    def translate_to_chinese(self):
        translate_to_chinese(self)

