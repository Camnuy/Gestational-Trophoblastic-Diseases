import os
import sys
import json
from PyQt5.QtWidgets import (QApplication, QMessageBox, QAction, QGraphicsPixmapItem, QMenu,
                             QWidget,QFileDialog, QHBoxLayout,QVBoxLayout,QTabWidget,QLabel,
                             QTextEdit,QDialog,QTreeWidgetItem)
from PyQt5.QtGui import QPixmap,QPainter,QTransform,QIcon
from PyQt5.QtCore import Qt, QRectF
from PIL import Image, ImageFile
#import QTimer
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images

def load_data(self):
    if not os.path.exists(self.data_folder):
        QMessageBox.critical(self, 'Fail', f'Data folder "{self.data_folder}" Not found')
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
                    patient_item = QTreeWidgetItem([f"{patient_data['name']}({patient_data['patient_index']})"])
                    
                    patient_item.setIcon(0, QIcon("res/patient_2.png"))  # 设置母目录图标
                    # Add subdirectories as slices
                    for sub_folder in os.listdir(patient_path):
                        sub_folder_path = os.path.join(patient_path, sub_folder)
                        if os.path.isdir(sub_folder_path) and sub_folder != "info.json":
                            slice_item = QTreeWidgetItem([sub_folder])
                            slice_item.setIcon(0, QIcon("res/slice.png"))  # 设置子目录图标
                            patient_item.setIcon(0, QIcon("res/patient.png"))  # 设置母目录图标
                            patient_item.addChild(slice_item)

                    self.patient_list.addTopLevelItem(patient_item)
            else:
                print(f"Not found {patient_folder}")
    

def load_patient_data(self):
    selected_items = self.patient_list.selectedItems()
    if selected_items:
        selected_item = selected_items[0]
        parent_item = selected_item.parent()
        # Collapse all other items
        root = self.patient_list.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item != selected_item and item != parent_item:
                item.setExpanded(False)

        if parent_item is None:
            # Clicked on a patient
            self.radio_button_slices.setChecked(True)  # 自动切换模式
            self.right_widget.setMinimumWidth(300)
            self.initialize_right_panel()

            index = self.patient_list.indexOfTopLevelItem(selected_item)
            self.current_patient_data = self.patients[index]

            self.patient_index_edit.setText(str(self.current_patient_data['patient_index']))
            self.name_edit.setText(self.current_patient_data['name'])
            self.gender_edit.setText(self.current_patient_data['gender'])
            self.age_edit.setText(str(self.current_patient_data['age']))
            self.department_edit.setText(self.current_patient_data['department'])
            self.submission_time_edit.setText(self.current_patient_data['submission_time'])
            self.admission_number_edit.setText(self.current_patient_data['admission_number'])
            self.bed_number_edit.setText(self.current_patient_data['bed_number'])
            self.doctor_edit.setText(self.current_patient_data['doctor'])
            self.result_edit.setText(self.current_patient_data['result'])
            self.report_text.setText(self.current_patient_data['diagnosis_report'])

            patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])
            self.image_folders = [name for name in os.listdir(patient_folder) if os.path.isdir(os.path.join(patient_folder, name))]
            if self.image_folders:
                self.current_image_index = 0
                self.message_box = QMessageBox()
                self.message_box.setWindowTitle('Please wait')
                self.message_box.setText('Loading...')
                self.message_box.setStandardButtons(QMessageBox.NoButton)  # 不显示任何按钮
                self.message_box.setWindowModality(Qt.WindowModal)  # 设置为窗口模态
                self.message_box.setWindowFlags(self.message_box.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                self.message_box.show()
                QApplication.processEvents()  # 处理所有挂起的事件
                self.display_image_top(self.image_folders[0] + '/' + self.image_folders[0] + '_resized.png')
                self.message_box.hide()
            else:
                self.graphics_view_top.scene.clear()
                # QMessageBox.warning(self, '警告', f'未找到病理切片文件 {self.current_patient_data["name"]}')
                print(f'Not found {self.current_patient_data["name"]}')
            self.display_image_lesion()
            # Expand the selected patient item
            selected_item.setExpanded(True)
        else:
            # Clicked on a slice
            patient_index = self.patient_list.indexOfTopLevelItem(parent_item)
            patient_data = self.patients[patient_index]
            slice_name = selected_item.text(0)
            patient_folder = os.path.join(self.data_folder, patient_data['folder'])
            slice_folder = os.path.join(patient_folder, slice_name)
            slice_image_relative_path = f'{slice_name}/{slice_name}_resized.png'
            slice_image_path = os.path.join(slice_folder, f'{slice_name}_resized.png')
            if os.path.exists(slice_image_path):
                self.display_image_top(slice_image_relative_path)
            else:
                # QMessageBox.warning(self, '警告', f'未找到病理切片文件 {slice_name}')
                print(f'Not found {slice_name}')

# 新增的 on_item_expanded 方法
def on_item_expanded(self, item):
    root = self.patient_list.invisibleRootItem()
    for i in range(root.childCount()):
        child = root.child(i)
        if child != item:
            child.setExpanded(False)


def display_image_top(self, image_file):
    # 获取图像路径
    patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])
    image_path = os.path.join(patient_folder, image_file)
    self.rongmao_path = image_path[:-4] + "_rongmao.png"
    self.shuizhong_path = image_path[:-4] + "_shuizhong.png"
    self.zengsheng_path = image_path[:-4] + "_zengsheng.png"

    # 检查图像文件是否存在
    if os.path.exists(image_path):
        # 加载图像
        self.image_top = QPixmap(image_path)
        print(f"Loaded image with size: {self.image_top.size()}")

        # 使用 set_original_image 方法来设置和显示原始图像
        self.graphics_view_top.set_original_image(self.image_top)
        print("Set original image in graphics view top")

        # 确保应用缩放
        self.zoom_image_top()

        # 更新当前图像路径
        self.current_image_path = image_path

        # 清空底部图像显示和 overlay
        self.graphics_view_bottom.clear_overlays()
        self.graphics_view_bottom.scene.clear()
        self.graphics_view_bottom.original_image_item = None  # 确保清除原始图像项

        # 将复选框状态设置为未选中
        self.rongmao_checkbox.setChecked(False)
        self.shuizhong_checkbox.setChecked(False)
        self.zengsheng_checkbox.setChecked(False)

        print("Cleared bottom scene and reset checkboxes")
    else:
        self.graphics_view_top.scene.clear()
        self.image_item_top = None
        print(f'Not found: {image_file}')




def display_image_lesion(self):
    # 获取当前患者文件夹路径
    patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])

    # 加载第一张图像
    image_path1 = os.path.join(patient_folder, 'lesion_image_1.png')
    print(image_path1)
    pixmap1 = QPixmap(image_path1)
    if not pixmap1.isNull():
        thumbnail1 = pixmap1.scaled(140, 140, Qt.KeepAspectRatio)
        self.image1.setPixmap(thumbnail1)
        self.image1.repaint()  # 强制刷新
        self.image1.mousePressEvent = lambda event: self.show_full_image(image_path1)
    else:
        # QMessageBox.warning(self, '警告', f'未找到病灶截图文件: {image_path1}')
        print(f'Not found: {image_path1}')

    # 加载第二张图像
    image_path2 = os.path.join(patient_folder, 'lesion_image_2.png')
    pixmap2 = QPixmap(image_path2)
    if not pixmap2.isNull():
        thumbnail2 = pixmap2.scaled(140, 140, Qt.KeepAspectRatio)
        self.image2.setPixmap(thumbnail2)
        self.image2.repaint()
        self.image2.mousePressEvent = lambda event: self.show_full_image(image_path2)
    else:
        # QMessageBox.warning(self, '警告', f'未找到病灶截图文件: {image_path2}')
        print(f'Not found: {image_path2}')

def show_full_image(self, image_path):
    dialog = QDialog(self)
    dialog.setWindowTitle("Image")
    dialog.setModal(True)
    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # 去掉问号图标
    layout = QVBoxLayout(dialog)
    label = QLabel(dialog)
    pixmap = QPixmap(image_path)
    label.setPixmap(pixmap)
    layout.addWidget(label)
    dialog.exec_()


def take_screenshot(self):
    # 获取当前患者文件夹路径
    patient_folder = os.path.join(self.data_folder, self.current_patient_data['folder'])

    # 定义固定文件名
    lesion_image_1 = os.path.join(patient_folder, 'lesion_image_1.png')
    lesion_image_2 = os.path.join(patient_folder, 'lesion_image_2.png')

    # 检查现有的病灶图像
    if os.path.exists(lesion_image_1) and os.path.exists(lesion_image_2):
        # 如果两张图像都存在，覆盖最早保存的那张图
        if os.path.getmtime(lesion_image_1) < os.path.getmtime(lesion_image_2):
            save_path = lesion_image_1
        else:
            save_path = lesion_image_2
    elif os.path.exists(lesion_image_1):
        save_path = lesion_image_2
    else:
        save_path = lesion_image_1

    # 截图
    rect = self.graphics_view_bottom.viewport().rect()
    screenshot = QPixmap(rect.size())
    painter = QPainter(screenshot)
    self.graphics_view_bottom.render(painter, QRectF(screenshot.rect()), rect)
    painter.end()

    # 保存截图
    screenshot.save(save_path)
    
    # 显示保存成功的提示框
    QMessageBox.information(self, 'Save', f'Save path: {save_path}')

    self.display_image_lesion()

def zoom_image_top(self):
    scale_factor = self.zoom_slider_top.value() / 100.0
    factor = scale_factor / self.graphics_view_top.zoom_factor
    self.graphics_view_top.setTransform(QTransform.fromScale(factor, factor), True)
    self.graphics_view_top.zoom_factor = scale_factor

def zoom_image_bottom(self):
    scale_factor = self.zoom_slider_bottom.value() / 100.0
    factor = scale_factor / self.graphics_view_bottom.zoom_factor
    self.graphics_view_bottom.setTransform(QTransform.fromScale(factor, factor), True)
    self.graphics_view_bottom.zoom_factor = scale_factor

def update_slider_value(self, view):
    if view == self.graphics_view_top:
        self.zoom_slider_top.setValue(int(view.zoom_factor * 100))
    elif view == self.graphics_view_bottom:
        self.zoom_slider_bottom.setValue(int(view.zoom_factor * 100))

def show_previous_image(self):
    if self.image_folders and self.current_image_index > 0:
        self.current_image_index -= 1
        self.display_image_top(self.image_folders[self.current_image_index]+'/'+self.image_folders[self.current_image_index]+'_resized.png')

def show_next_image(self):
    if self.image_folders and self.current_image_index < len(self.image_folders) - 1:
        self.current_image_index += 1
        self.display_image_top(self.image_folders[self.current_image_index]+'/'+self.image_folders[self.current_image_index]+'_resized.png')

def show_context_menu(self, pos):
    context_menu = QMenu(self)
    delete_action = QAction("Delete", self)
    delete_action.triggered.connect(self.delete_patient)
    context_menu.addAction(delete_action)
    context_menu.exec_(self.patient_list.viewport().mapToGlobal(pos))


def initialize_right_panel(self):
    # 清空 tab_widget 的内容
    if hasattr(self, 'tab_widget'):
        self.tab_widget.clear()
    else:
        self.tab_widget = QTabWidget()
        self.right_layout.addWidget(self.tab_widget)
    
    # Canvas widget tab
    self.canvas_widget = QWidget()
    canvas_widget_tab = QWidget()
    canvas_widget_layout = QVBoxLayout(canvas_widget_tab)
    canvas_widget_layout.addWidget(self.canvas_widget)
    canvas_widget_tab.setLayout(canvas_widget_layout)
    self.tab_widget.addTab(canvas_widget_tab, "Pie Chart")
    
    # Lesion images tab
    lesion_images_tab = QWidget()
    lesion_images_layout = QVBoxLayout(lesion_images_tab)
    
    # Adding two images to the layout
    self.image_layout = QHBoxLayout()
    
    if not hasattr(self, 'image1'):
        self.image1 = QLabel()
    pixmap1 = QPixmap()  # Initial empty pixmap
    self.image1.setPixmap(pixmap1)
    self.image_layout.addWidget(self.image1)
    
    if not hasattr(self, 'image2'):
        self.image2 = QLabel()
    pixmap2 = QPixmap()  # Initial empty pixmap
    self.image2.setPixmap(pixmap2)
    self.image_layout.addWidget(self.image2)
    
    lesion_images_layout.addLayout(self.image_layout)
    lesion_images_tab.setLayout(lesion_images_layout)
    self.tab_widget.addTab(lesion_images_tab, "Lesion")
    
    # Detailed info tab
    self.detailed_info_text = QTextEdit()
    detailed_info_tab = QWidget()
    detailed_info_layout = QVBoxLayout(detailed_info_tab)
    detailed_info_layout.addWidget(self.detailed_info_text)
    detailed_info_tab.setLayout(detailed_info_layout)
    self.tab_widget.addTab(detailed_info_tab, "Statistics")
