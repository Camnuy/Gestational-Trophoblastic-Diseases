
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QVBoxLayout, QPushButton, QApplication, QLabel, QHBoxLayout, QRadioButton, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QPainterPath
from PyQt5.QtCore import Qt, QPoint
import requests
import numpy as np
from io import BytesIO
import os
import cv2
from PyQt5.QtCore import QRectF
from PIL import Image, ImageDraw, ImageFile
import time
#import QTimer





Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images
def run_segmentation_online(self,index = 0):
    if index == 0:
        is_rongmao_checked = self.rongmao_checkbox.isChecked()
        is_shuizhong_checked = self.shuizhong_checkbox.isChecked()
        is_zengsheng_checked = self.zengsheng_checkbox.isChecked()
        print('ok')

        selected_items = self.patient_list.selectedItems()
        if selected_items and self.current_image_path:
            image_path = self.current_image_path

            if os.path.exists(image_path):
                if self.graphics_view_bottom.original_image_item is None:
                    self.display_image_bottom(image_path)

                if is_rongmao_checked:
                    if not self.graphics_view_bottom.overlay_exists(1):
                        self.add_overlay_bottom(image_path, self.rongmao_path, 1)
                        if not os.path.isfile(self.rongmao_path[:-4]+'_diagnosis.png'):
                            overlay_image = upload_image(self, image_path, 0)
                            overlay_image.save(self.rongmao_path[:-4]+'_diagnosis.png', format="PNG")
                else:
                    self.graphics_view_bottom.remove_overlay(1)

                if is_shuizhong_checked:
                    if not self.graphics_view_bottom.overlay_exists(2):
                        self.add_overlay_bottom(image_path, self.shuizhong_path, 2)
                else:
                    self.graphics_view_bottom.remove_overlay(2)

                if is_zengsheng_checked:
                    if not self.graphics_view_bottom.overlay_exists(3):
                        self.add_overlay_bottom(image_path, self.zengsheng_path, 3)
                else:
                    self.graphics_view_bottom.remove_overlay(3)
            else:
                print("Image path does not exist.")
    else:
        is_rongmao_checked = True
        is_shuizhong_checked = True
        is_zengsheng_checked = True

        selected_items = self.patient_list.selectedItems()
        if selected_items and self.current_image_path:
            image_path = self.current_image_path

            if os.path.exists(image_path):
                if self.graphics_view_bottom.original_image_item is None:
                    self.display_image_bottom(image_path)

                if is_rongmao_checked:
                    if not self.graphics_view_bottom.overlay_exists(1):
                        self.add_overlay_bottom(image_path, self.rongmao_path, 1)
                        if not os.path.isfile(self.rongmao_path[:-4]+'_diagnosis.png'):
                            overlay_image = upload_image(self, image_path, 0)
                            overlay_image.save(self.rongmao_path[:-4]+'_diagnosis.png', format="PNG")
                else:
                    self.graphics_view_bottom.remove_overlay(1)

                if is_shuizhong_checked:
                    if not self.graphics_view_bottom.overlay_exists(2):
                        self.add_overlay_bottom(image_path, self.shuizhong_path, 2)
                else:
                    self.graphics_view_bottom.remove_overlay(2)

                if is_zengsheng_checked:
                    if not self.graphics_view_bottom.overlay_exists(3):
                        self.add_overlay_bottom(image_path, self.zengsheng_path, 3)
                else:
                    self.graphics_view_bottom.remove_overlay(3)
            else:
                print("Image path does not exist.")

def display_image_bottom(self, image_path):
    if os.path.exists(image_path):
        self.image_bottom = QPixmap(image_path)
        self.graphics_view_bottom.set_original_image(self.image_bottom)

def add_overlay_bottom(self, image_path, overlay_path, overlay_type):
    if not os.path.isfile(overlay_path):
        overlay_image = upload_image(self, image_path, overlay_type)
        overlay_image.save(overlay_path, format="PNG")

    segmentation_result = QImage(overlay_path)
    overlay_pixmap = QPixmap.fromImage(segmentation_result)
    if self.graphics_view_bottom.original_image_item:
        if overlay_pixmap.size() != self.graphics_view_bottom.original_image_item.pixmap().size():
            overlay_pixmap = overlay_pixmap.scaled(self.graphics_view_bottom.original_image_item.pixmap().size())
        overlay_pixmap.toImage().setText("overlay_path", str(overlay_type))  # 使用字符串表示叠加层类型
        self.graphics_view_bottom.add_overlay(overlay_pixmap, overlay_type)  # 使用半透明红色
    else:
        print("Original image item does not exist.")



def upload_image(self, file_path, index):
    if file_path:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'index': index,  # or 2, or 3 depending on your use case
                'patch_size': 512,
                'step_size': None  # or a specific value if needed
            }
            response = requests.post('http://127.0.0.1:5000/upload', files=files, data=data)
            if response.status_code == 200:
                print("Image processing completed and saved")

                pre_label_img = Image.open(BytesIO(response.content))
                return pre_label_img
            
            else:
                print(f"Failed to upload image: {response.status_code}")


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.current_tool = 'brush'
        self.current_color = QColor(255, 0, 0, 150)  # Default to red with transparency
        self.drawing = False
        self.last_point = QPoint()
        self.overlay_images = []  # Initialize as a list to store multiple overlay images
        self.overlay_items = []  # Initialize as a list to store multiple overlay items

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = self.mapToScene(event.pos()).toPoint()

    def mouseMoveEvent(self, event):
        if self.drawing:
            for overlay_image, overlay_item in zip(self.overlay_images, self.overlay_items):
                painter = QPainter(overlay_image)
                if self.current_tool == 'brush':
                    pen = QPen(self.current_color, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # Thicker brush
                    painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                elif self.current_tool == 'eraser':
                    pen = QPen(Qt.transparent, 40, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # Larger eraser
                    painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.setPen(pen)
                painter.drawLine(self.last_point, self.mapToScene(event.pos()).toPoint())
                overlay_item.setPixmap(QPixmap.fromImage(overlay_image))
            self.last_point = self.mapToScene(event.pos()).toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def set_tool(self, tool):
        self.current_tool = tool

    def set_color(self, color):
        self.current_color = color

    def add_overlay_image(self, overlay_image):
        self.overlay_images.append(overlay_image)
        overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(overlay_image))
        self.overlay_items.append(overlay_item)
        self.scene().addItem(overlay_item)

def online_learn(self):
    if self.current_image_path:
        image_path = self.current_image_path
        rongmao_path = self.rongmao_path
        shuizhong_path = self.shuizhong_path
        zengsheng_path = self.zengsheng_path
        if os.path.exists(image_path):
            # Create a new dialog to show the image
            dialog = QDialog(self)
            dialog.setWindowTitle("picture")
            dialog.resize(800, 600)
            dialog_layout = QVBoxLayout(dialog)

            # Create a QGraphicsView to show the image
            self.graphics_view = ZoomableGraphicsView(dialog)
            graphics_scene = QGraphicsScene(self.graphics_view)
            self.graphics_view.setScene(graphics_scene)

            # Load and set the background image
            pixmap = QPixmap(image_path)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            graphics_scene.addItem(pixmap_item)

            # Add overlays on top of the background image
            self.add_overlay(graphics_scene, rongmao_path, QColor(0, 0, 255, 100))  # Blue overlay for Rongmao
            self.add_overlay(graphics_scene, shuizhong_path, QColor(0, 255, 0, 100))  # Green overlay for Shuizhong
            self.add_overlay(graphics_scene, zengsheng_path, QColor(255, 0, 0, 100))  # Red overlay for Zengsheng

            # Add button to dialog layout
            button = QPushButton("get into online_learn dataset")
            button.clicked.connect(self.show_added_to_database_dialog)

            # Add tool selection buttons
            tool_layout = QHBoxLayout()
            brush_button = QPushButton("brush")
            brush_button.clicked.connect(lambda: self.graphics_view.set_tool('brush'))
            eraser_button = QPushButton("ereaser")
            eraser_button.clicked.connect(lambda: self.graphics_view.set_tool('eraser'))
            tool_layout.addWidget(brush_button)
            tool_layout.addWidget(eraser_button)

            # Add color selection radio buttons
            color_layout = QHBoxLayout()
            self.color_group = QButtonGroup()
            self.color_group.buttonClicked.connect(self.change_color)
            red_radio = QRadioButton("绒毛 (red)")
            red_radio.setChecked(True)
            green_radio = QRadioButton("水肿 (green)")
            blue_radio = QRadioButton("增生 (blue)")
            self.color_group.addButton(red_radio, 1)
            self.color_group.addButton(green_radio, 2)
            self.color_group.addButton(blue_radio, 3)
            color_layout.addWidget(red_radio)
            color_layout.addWidget(green_radio)
            color_layout.addWidget(blue_radio)

            dialog_layout.addWidget(self.graphics_view)
            dialog_layout.addLayout(tool_layout)
            dialog_layout.addLayout(color_layout)
            dialog_layout.addWidget(button)
            dialog.setLayout(dialog_layout)
            dialog.exec_()

def show_added_to_database_dialog(self):
    # Show a dialog indicating the image was added to the online learning database
    message_dialog = QDialog(self)
    message_dialog.setWindowTitle("information")
    message_layout = QVBoxLayout(message_dialog)
    message_label = QLabel("Get into online_learn dataset")
    message_layout.addWidget(message_label)
    message_dialog.setLayout(message_layout)
    message_dialog.exec_()

def add_overlay(self, scene, overlay_path, color):
    if os.path.exists(overlay_path):
        overlay_image = QImage(overlay_path).convertToFormat(QImage.Format_ARGB32)
        width, height = overlay_image.width(), overlay_image.height()
        ptr = overlay_image.bits()
        ptr.setsize(overlay_image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)

        # Create a mask for white pixels
        mask = np.all(arr[:, :, :3] == [255, 255, 255], axis=-1)
        arr[mask, :3] = [color.red(), color.green(), color.blue()]
        arr[mask, 3] = color.alpha()
        arr[~mask, 3] = 0

        overlay_image = QImage(arr.data, width, height, QImage.Format_ARGB32)
        self.graphics_view.add_overlay_image(overlay_image)

def change_color(self, button):
    color_map = {1: QColor(255, 0, 0, 150), 2: QColor(0, 255, 0, 150), 3: QColor(0, 0, 255, 150)}
    self.graphics_view.set_color(color_map[self.color_group.id(button)])