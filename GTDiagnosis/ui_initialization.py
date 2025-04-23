from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
                             QTextEdit, QSlider, QPushButton, QLineEdit, 
                             QFormLayout, QToolBar, QAction, QGraphicsView, 
                             QGraphicsScene, QSplitter, QComboBox,QSizePolicy,
                              QFileDialog,QTabWidget, QCheckBox,QTreeWidget,QGraphicsPixmapItem )
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QRectF, QPointF, QUrl
from PyQt5.QtWidgets import QRadioButton, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
import numpy as np
import cv2

class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.original_image_item = None
        self.overlay_items = {}

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom_in(self):
        self.zoom_factor *= 1.1
        self.scale(1.1, 1.1)

    def zoom_out(self):
        self.zoom_factor /= 1.1
        self.scale(1 / 1.1, 1 / 1.1)

    def set_original_image(self, pixmap):
        self.scene.clear()
        self.original_image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.original_image_item)
        self.scene.setSceneRect(QRectF(pixmap.rect()).adjusted(-500, -500, 500, 500))
        self.fitInView(self.original_image_item, Qt.KeepAspectRatio)
        self.scene.update()
        self.update()

    def add_overlay(self, pixmap, overlay_type):
        if overlay_type == 1:
            color = QColor(0, 0, 255, 150)
        elif overlay_type == 2:
            color = QColor(0, 255, 0, 150)
        elif overlay_type == 3:
            color = QColor(255, 0, 0, 150)
        else:
            color = QColor(255, 255, 255, 150)

        image = pixmap.toImage()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)

        overlay_arr = np.zeros_like(arr)
        mask = (arr[:, :, 0:3] == [255, 255, 255]).all(axis=2)
        overlay_arr[mask, 0:3] = [color.red(), color.green(), color.blue()]
        overlay_arr[mask, 3] = color.alpha()

        colored_image = QImage(overlay_arr.data, overlay_arr.shape[1], overlay_arr.shape[0], QImage.Format_ARGB32)
        colored_pixmap = QPixmap.fromImage(colored_image)
        overlay_item = QGraphicsPixmapItem(colored_pixmap)
        overlay_item.setOpacity(0.6)
        overlay_item.setZValue(1)
        overlay_item.setPos(self.original_image_item.pos() if self.original_image_item else QPointF(0, 0))
        self.overlay_items[overlay_type] = overlay_item
        self.scene.addItem(overlay_item)

    def remove_overlay(self, overlay_type):
        if overlay_type in self.overlay_items:
            item = self.overlay_items.pop(overlay_type)
            self.scene.removeItem(item)

    def clear_overlays(self):
        while self.overlay_items:
            overlay_type, item = self.overlay_items.popitem()
            self.scene.removeItem(item)

    def overlay_exists(self, overlay_type):
        return overlay_type in self.overlay_items

class CustomTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super(CustomTreeWidget, self).__init__(parent)
        self.setIndentation(10)

    def drawBranches(self, painter, rect, index):
        pass

    def drawRow(self, painter, option, index):
        option.rect.setLeft(option.rect.left() - 20)
        super(CustomTreeWidget, self).drawRow(painter, option, index)

def initUI(self):
    self.setWindowTitle('病理诊断软件')

    self.main_widget = QWidget()
    self.main_layout = QVBoxLayout(self.main_widget)
    self.main_layout.setContentsMargins(0, 0, 0, 0)

    self.top_bar = QWidget(objectName="top_bar")
    self.top_bar_layout = QHBoxLayout(self.top_bar)
    self.top_bar_layout.setContentsMargins(0, 0, 0, 0)

    self.logo_label = QLabel()
    logo_pixmap = QPixmap("res/logo.png")
    self.logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio))
    self.top_bar_layout.addWidget(self.logo_label, alignment=Qt.AlignLeft)

    self.top_bar_layout.addStretch()

    self.language_combo = QComboBox()
    self.language_combo.addItem("English")
    self.language_combo.addItem("中文")
    self.language_combo.currentIndexChanged.connect(self.change_language)
    self.top_bar_layout.addWidget(self.language_combo, alignment=Qt.AlignRight)

    # Add help button
    self.help_button = QPushButton("帮助")
    self.help_button.clicked.connect(self.open_help_pdf)
    self.top_bar_layout.addWidget(self.help_button, alignment=Qt.AlignRight)

    # Add developer info button
    self.dev_info_button = QPushButton("开发者信息")
    self.dev_info_button.clicked.connect(self.open_dev_info_pdf)
    self.top_bar_layout.addWidget(self.dev_info_button, alignment=Qt.AlignRight)

    self.close_button = QPushButton("×", self)
    self.close_button.clicked.connect(self.close)
    self.top_bar_layout.addWidget(self.close_button, alignment=Qt.AlignRight)

    self.top_bar.setFixedHeight(70)
    self.main_layout.addWidget(self.top_bar)

    horizontal_splitter = QSplitter(Qt.Horizontal)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)

    self.radio_button_microscope = QRadioButton("显微镜模式")
    self.radio_button_slices = QRadioButton("切片扫描模式")
    self.radio_button_slices.setChecked(True)
    self.mode_selection_group = QButtonGroup()
    self.mode_selection_group.addButton(self.radio_button_microscope)
    self.mode_selection_group.addButton(self.radio_button_slices)
    self.mode_selection_group.buttonClicked.connect(self.change_mode)
    left_layout.addWidget(self.radio_button_microscope)
    left_layout.addWidget(self.radio_button_slices)

    self.add_patient_button = QPushButton("添加新患者")
    self.add_patient_button.clicked.connect(self.add_new_patient)
    left_layout.addWidget(self.add_patient_button)

    self.add_patient_group_button = QPushButton("批量添加新患者")
    self.add_patient_group_button.clicked.connect(self.add_new_patient_group)
    left_layout.addWidget(self.add_patient_group_button)

    self.import_image_button = QPushButton("添加新切片")
    self.import_image_button.clicked.connect(self.import_new_image)
    left_layout.addWidget(self.import_image_button)

    self.import_image_group_button = QPushButton("批量添加新切片")
    self.import_image_group_button.clicked.connect(self.import_new_image_group)
    left_layout.addWidget(self.import_image_group_button)

    self.camera_selection_combo = QComboBox()
    self.camera_selection_combo.addItem("选择摄像头")
    available_cameras = list_available_cameras()
    for cam_index in available_cameras:
        self.camera_selection_combo.addItem(f"摄像头 {cam_index}", cam_index)
    self.camera_selection_combo.currentIndexChanged.connect(self.select_camera_index)
    left_layout.addWidget(self.camera_selection_combo)

    self.patient_list = CustomTreeWidget()
    self.patient_list.setHeaderHidden(True)
    self.patient_list.setContextMenuPolicy(Qt.CustomContextMenu)
    self.patient_list.customContextMenuRequested.connect(self.show_context_menu)
    self.patient_list.itemSelectionChanged.connect(self.load_patient_data)
    self.patient_list.itemExpanded.connect(self.on_item_expanded)

    left_layout.addWidget(self.patient_list)
    left_widget.setLayout(left_layout)
    left_widget.setMinimumWidth(250)
    horizontal_splitter.addWidget(left_widget)

    middle_widget = QWidget()
    middle_layout = QHBoxLayout(middle_widget)

    self.prev_button = QPushButton("<")
    self.prev_button.setObjectName("prevButton")
    self.prev_button.clicked.connect(self.show_previous_image)
    self.prev_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    middle_layout.addWidget(self.prev_button)

    graphics_splitter = QSplitter(Qt.Horizontal)

    left_image_widget = QWidget()
    left_image_layout = QVBoxLayout(left_image_widget)

    self.graphics_view_top = CustomGraphicsView(left_image_widget)
    self.graphics_view_top.setDragMode(QGraphicsView.ScrollHandDrag)
    left_image_layout.addWidget(self.graphics_view_top)

    self.zoom_slider_top = QSlider(Qt.Horizontal)
    self.zoom_slider_top.setMinimum(10)
    self.zoom_slider_top.setMaximum(200)
    self.zoom_slider_top.setValue(10)
    self.zoom_slider_top.valueChanged.connect(self.zoom_image_top)
    left_image_layout.addWidget(self.zoom_slider_top)

    left_image_widget.setLayout(left_image_layout)
    graphics_splitter.addWidget(left_image_widget)

    right_image_widget = QWidget()
    right_image_layout = QVBoxLayout(right_image_widget)

    self.graphics_view_bottom = CustomGraphicsView(right_image_widget)
    self.graphics_view_bottom.setDragMode(QGraphicsView.ScrollHandDrag)
    right_image_layout.addWidget(self.graphics_view_bottom)

    self.zoom_slider_bottom = QSlider(Qt.Horizontal)
    self.zoom_slider_bottom.setMinimum(10)
    self.zoom_slider_bottom.setMaximum(200)
    self.zoom_slider_bottom.setValue(10)
    self.zoom_slider_bottom.valueChanged.connect(self.zoom_image_bottom)
    right_image_layout.addWidget(self.zoom_slider_bottom)

    self.screenshot_button = QPushButton("截图获取报告病灶图像")
    self.screenshot_button.clicked.connect(self.take_screenshot)
    right_image_layout.addWidget(self.screenshot_button)

    self.checkbox_widget = QWidget()
    checkbox_layout = QVBoxLayout()

    checkbox_with_control_layout = QHBoxLayout()
    checkbox_with_control_layout.addLayout(checkbox_layout)

    self.online_learn_button = QPushButton("在线学习标注")
    self.online_learn_button.clicked.connect(self.online_learn) 
    checkbox_with_control_layout.addWidget(self.online_learn_button)

    self.rongmao_checkbox = QCheckBox("AI智能绒毛识别")
    self.shuizhong_checkbox = QCheckBox("AI智能水肿识别")
    self.zengsheng_checkbox = QCheckBox("AI智能增生识别")

    self.rongmao_checkbox.stateChanged.connect(lambda state: self.run_segmentation_online(index=0))
    self.shuizhong_checkbox.stateChanged.connect(lambda state: self.run_segmentation_online(index=0))
    self.zengsheng_checkbox.stateChanged.connect(lambda state: self.run_segmentation_online(index=0))

    checkbox_layout.addWidget(self.rongmao_checkbox)
    checkbox_layout.addWidget(self.shuizhong_checkbox)
    checkbox_layout.addWidget(self.zengsheng_checkbox)

    self.checkbox_widget.setLayout(checkbox_with_control_layout)
    right_image_layout.addWidget(self.checkbox_widget, alignment=Qt.AlignBottom)

    self.toggle_and_clear_layout = QHBoxLayout()
    self.toggle_segmentation_button = QCheckBox("打开AI智能识别")
    self.toggle_segmentation_button.setChecked(False)
    self.toggle_segmentation_button.stateChanged.connect(self.toggle_segmentation)
    self.toggle_and_clear_layout.addWidget(self.toggle_segmentation_button)

    self.clear_button = QPushButton("清空")
    self.clear_button.clicked.connect(self.clear_right_image)
    self.toggle_and_clear_layout.addWidget(self.clear_button)

    self.toggle_and_clear_widget = QWidget()
    self.toggle_and_clear_widget.setLayout(self.toggle_and_clear_layout)
    right_image_layout.addWidget(self.toggle_and_clear_widget, alignment=Qt.AlignBottom)

    self.checkbox_widget.setVisible(True)
    self.toggle_and_clear_widget.setVisible(False)

    right_image_widget.setLayout(right_image_layout)
    graphics_splitter.addWidget(right_image_widget)

    middle_layout.addWidget(graphics_splitter)

    self.next_button = QPushButton(">")
    self.next_button.setObjectName("nextButton")
    self.next_button.clicked.connect(self.show_next_image)
    self.next_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    middle_layout.addWidget(self.next_button)

    middle_widget.setLayout(middle_layout)
    horizontal_splitter.addWidget(middle_widget)

    self.right_widget = QWidget()
    self.right_layout = QVBoxLayout(self.right_widget)

    self.info_form_layout = QFormLayout()
    self.info_form_layout.addRow(QLabel("患者基本信息"))

    self.patient_index_edit = QLineEdit()
    self.name_edit = QLineEdit()
    self.gender_edit = QLineEdit()
    self.age_edit = QLineEdit()
    self.department_edit = QLineEdit()
    self.submission_time_edit = QLineEdit()
    self.admission_number_edit = QLineEdit()
    self.bed_number_edit = QLineEdit()
    self.doctor_edit = QLineEdit()
    self.result_edit = QLineEdit()

    self.info_form_layout.addRow("病理号:", self.patient_index_edit)
    self.info_form_layout.addRow("姓名:", self.name_edit)
    self.info_form_layout.addRow("性别:", self.gender_edit)
    self.info_form_layout.addRow("年龄:", self.age_edit)
    self.info_form_layout.addRow("送检科室:", self.department_edit)
    self.info_form_layout.addRow("送检日期:", self.submission_time_edit)
    self.info_form_layout.addRow("住院号:", self.admission_number_edit)
    self.info_form_layout.addRow("床号:", self.bed_number_edit)
    self.info_form_layout.addRow("送检医生:", self.doctor_edit)
    self.info_form_layout.addRow("临床诊断:", self.result_edit)

    self.right_layout.addLayout(self.info_form_layout)

    self.update_info_button = QPushButton("更新基本信息")
    self.update_info_button.clicked.connect(self.update_patient_info)
    self.right_layout.addWidget(self.update_info_button)

    self.diagnosis_button = QPushButton("AI智能诊断")
    self.diagnosis_button.clicked.connect(self.diagnosis)
    self.right_layout.addWidget(self.diagnosis_button)

    self.tab_widget = QTabWidget()
    self.right_layout.addWidget(self.tab_widget)

    self.canvas_widget = QWidget()
    canvas_widget_tab = QWidget()
    canvas_widget_layout = QVBoxLayout(canvas_widget_tab)
    canvas_widget_layout.addWidget(self.canvas_widget)
    canvas_widget_tab.setLayout(canvas_widget_layout)
    self.tab_widget.addTab(canvas_widget_tab, "饼图")

    self.lesion_images_tab = QWidget()
    self.lesion_images_layout = QVBoxLayout(self.lesion_images_tab)

    self.image_layout = QHBoxLayout()

    self.image1 = QLabel()
    pixmap1 = QPixmap()
    self.image1.setPixmap(pixmap1)
    self.image_layout.addWidget(self.image1)

    self.image2 = QLabel()
    pixmap2 = QPixmap()
    self.image2.setPixmap(pixmap2)
    self.image_layout.addWidget(self.image2)

    self.lesion_images_layout.addLayout(self.image_layout)
    self.lesion_images_tab.setLayout(self.lesion_images_layout)
    self.tab_widget.addTab(self.lesion_images_tab, "病灶")

    self.detailed_info_text = QTextEdit()
    detailed_info_tab = QWidget()
    detailed_info_layout = QVBoxLayout(detailed_info_tab)
    detailed_info_layout.addWidget(self.detailed_info_text)
    detailed_info_tab.setLayout(detailed_info_layout)
    self.tab_widget.addTab(detailed_info_tab, "统计")

    self.report_text = QTextEdit()
    self.right_layout.addWidget(self.report_text)

    self.update_report_button = QPushButton("生成诊断报告")
    self.update_report_button.clicked.connect(self.update_diagnosis_report)
    self.right_layout.addWidget(self.update_report_button)

    self.right_widget.setLayout(self.right_layout)
    self.right_widget.setMinimumWidth(300)
    horizontal_splitter.addWidget(self.right_widget)

    horizontal_splitter.setStretchFactor(0, 1)
    horizontal_splitter.setStretchFactor(1, 8)
    horizontal_splitter.setStretchFactor(2, 1)

    self.main_layout.addWidget(horizontal_splitter)
    self.setCentralWidget(self.main_widget)

    self.data_folder = "data"
    self.image_item_top = None
    self.image_item_bottom = None
    self.current_image_index = 0
    self.current_image_path = None
    self.load_data()
    self.change_language()

def open_help_pdf(self):
    QDesktopServices.openUrl(QUrl.fromLocalFile("res/help.pdf"))

def open_dev_info_pdf(self):
    QDesktopServices.openUrl(QUrl.fromLocalFile("res/developers.pdf"))

def list_available_cameras():
    available_cameras = []
    for i in range(10):  # 假设最多有10个摄像头
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def select_camera(self, index):
    camera_index = self.camera_selection_combo.itemData(index)
    print(f"选择的摄像头索引: {camera_index}")
    return camera_index

def change_language(self):
    current_language = self.language_combo.currentText()
    if current_language == "中文":
        self.translate_to_chinese()
    elif current_language == "English":
        self.translate_to_english()

def translate_to_chinese(self):
    # Translate UI components to Chinese
    self.setWindowTitle('病理诊断软件')
    self.help_button.setText("帮助")
    self.dev_info_button.setText("开发者信息")
    self.close_button.setText("×")
    self.radio_button_microscope.setText("显微镜模式")
    self.radio_button_slices.setText("切片扫描模式")
    self.add_patient_button.setText("添加新患者")
    self.add_patient_group_button.setText("批量添加新患者")
    self.import_image_button.setText("添加新切片")
    self.import_image_group_button.setText("批量添加新切片")
    self.camera_selection_combo.setItemText(0, "选择摄像头")
    self.prev_button.setText("<")
    self.next_button.setText(">")
    self.screenshot_button.setText("截图获取报告病灶图像")
    self.online_learn_button.setText("在线学习标注")
    self.rongmao_checkbox.setText("AI智能绒毛识别")
    self.shuizhong_checkbox.setText("AI智能水肿识别")
    self.zengsheng_checkbox.setText("AI智能增生识别")
    self.toggle_segmentation_button.setText("打开AI智能识别")
    self.clear_button.setText("清空")
    self.info_form_layout.labelForField(self.patient_index_edit).setText("病理号:")
    self.info_form_layout.labelForField(self.name_edit).setText("姓名:")
    self.info_form_layout.labelForField(self.gender_edit).setText("性别:")
    self.info_form_layout.labelForField(self.age_edit).setText("年龄:")
    self.info_form_layout.labelForField(self.department_edit).setText("送检科室:")
    self.info_form_layout.labelForField(self.submission_time_edit).setText("送检日期:")
    self.info_form_layout.labelForField(self.admission_number_edit).setText("住院号:")
    self.info_form_layout.labelForField(self.bed_number_edit).setText("床号:")
    self.info_form_layout.labelForField(self.doctor_edit).setText("送检医生:")
    self.info_form_layout.labelForField(self.result_edit).setText("临床诊断:")
    self.update_info_button.setText("更新基本信息")
    self.diagnosis_button.setText("AI智能诊断")
    self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_widget.widget(0)), "饼图")
    self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_widget.widget(1)), "病灶")
    self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_widget.widget(2)), "统计")
    self.update_report_button.setText("生成诊断报告")
    self.info_form_layout.itemAt(0).widget().setText("患者基本信息")

def translate_to_english(self):
    # Translate UI components to English
    self.setWindowTitle('Pathology Diagnosis Software')
    self.help_button.setText("Help")
    self.dev_info_button.setText("Developer Info")
    self.close_button.setText("×")
    self.radio_button_microscope.setText("Microscope Mode")
    self.radio_button_slices.setText("Slice Scanning Mode")
    self.add_patient_button.setText("Add New Patient")
    self.add_patient_group_button.setText("Add New Patient Group")
    self.import_image_button.setText("Import New Slice")
    self.import_image_group_button.setText("Import New Slice Group")
    self.camera_selection_combo.setItemText(0, "Select Camera")
    self.prev_button.setText("<")
    self.next_button.setText(">")
    self.screenshot_button.setText("Take Screenshot for Lesion Image")
    self.online_learn_button.setText("Online Learning Annotation")
    self.rongmao_checkbox.setText("AI Intelligent Villous Recognition")
    self.shuizhong_checkbox.setText("AI Intelligent Edema Recognition")
    self.zengsheng_checkbox.setText("AI Intelligent Hyperplasia Recognition")
    self.toggle_segmentation_button.setText("Enable AI Intelligent Recognition")
    self.clear_button.setText("Clear")
    self.info_form_layout.labelForField(self.patient_index_edit).setText("Pathology Number:")
    self.info_form_layout.labelForField(self.name_edit).setText("Name:")
    self.info_form_layout.labelForField(self.gender_edit).setText("Gender:")
    self.info_form_layout.labelForField(self.age_edit).setText("Age:")
    self.info_form_layout.labelForField(self.department_edit).setText("Department:")
    self.info_form_layout.labelForField(self.submission_time_edit).setText("Submission Date:")
    self.info_form_layout.labelForField(self.admission_number_edit).setText("Admission Number:")
    self.info_form_layout.labelForField(self.bed_number_edit).setText("Bed Number:")
    self.info_form_layout.labelForField(self.doctor_edit).setText("Doctor:")
    self.info_form_layout.labelForField(self.result_edit).setText("Clinical Diagnosis:")
    self.update_info_button.setText("Update Basic Information")
    self.diagnosis_button.setText("AI Intelligent Diagnosis")
    self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_widget.widget(0)), "Pie Chart")
    self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_widget.widget(1)), "Lesion")
    self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_widget.widget(2)), "Statistics")
    self.update_report_button.setText("Generate Diagnosis Report")
    self.info_form_layout.itemAt(0).widget().setText("Patient Information")