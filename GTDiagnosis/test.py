import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt5.QtCore import Qt
from PIL import Image
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
                             QTextEdit, QSlider, QPushButton, QLineEdit, 
                             QFormLayout, QToolBar, QAction, QGraphicsView, 
                             QGraphicsScene, QSplitter, QComboBox,QSizePolicy,
                              QFileDialog,QTabWidget, QCheckBox,QTreeWidget,QGraphicsPixmapItem )
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QRadioButton, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PIL import Image, ImageDraw

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np

class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.original_image_item = None
        self.overlay_items = []

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
        if self.original_image_item:
            self.scene.removeItem(self.original_image_item)
        self.original_image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.original_image_item)
        self.fitInView(self.original_image_item, Qt.KeepAspectRatio)  # 自动调整视图以适应图像

    def add_overlay(self, pixmap, color):
        print(f"Adding overlay with size: {pixmap.size()} and color: {color}")

        # 将 QPixmap 转换为 QImage
        image = pixmap.toImage()

        # 将 QImage 转换为 NumPy 数组
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)

        # 只对白色像素（标签层）进行颜色填充
        mask = (arr[:, :, 0:3] == [255, 255, 255]).all(axis=2)
        arr[mask, 0:3] = [color.red(), color.green(), color.blue()]
        arr[mask, 3] = color.alpha()

        # 将 NumPy 数组转换回 QImage
        colored_image = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format_ARGB32)

        # 将填充颜色后的 QImage 转换回 QPixmap
        colored_pixmap = QPixmap.fromImage(colored_image)

        overlay_item = QGraphicsPixmapItem(colored_pixmap)
        overlay_item.setOpacity(0.4)  # 调整透明度
        overlay_item.setZValue(1)  # 确保它在原始图像之上
        overlay_item.setPos(0, 0)
        self.overlay_items.append(overlay_item)
        self.scene.addItem(overlay_item)

    def clear_overlays(self):
        for item in self.overlay_items:
            self.scene.removeItem(item)
        self.overlay_items.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CustomGraphicsView Test")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.graphics_view = CustomGraphicsView(self)
        self.layout.addWidget(self.graphics_view)

        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        self.add_overlay_button = QPushButton("Add Overlay")
        self.add_overlay_button.clicked.connect(self.add_overlay)
        self.layout.addWidget(self.add_overlay_button)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.graphics_view.set_original_image(pixmap)

    def add_overlay(self):
        if not self.graphics_view.original_image_item:
            return

        # 选择叠加图像
        overlay_file_path, _ = QFileDialog.getOpenFileName(self, "Open Overlay Image File", "", "Images (*.png *.jpg *.bmp)")
        if not overlay_file_path:
            return

        # 创建叠加图像的 QPixmap
        overlay_pixmap = QPixmap(overlay_file_path)

        # 确保叠加图像与原始图像的尺寸一致
        if overlay_pixmap.size() != self.graphics_view.original_image_item.pixmap().size():
            overlay_pixmap = overlay_pixmap.scaled(self.graphics_view.original_image_item.pixmap().size())

        # 添加叠加图层
        self.graphics_view.add_overlay(overlay_pixmap, QColor(255, 0, 0, 100))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
