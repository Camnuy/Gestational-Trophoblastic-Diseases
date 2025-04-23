import os
from PyQt5.QtWidgets import (QGraphicsPixmapItem )
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRectF
from PIL import Image, ImageFile
import time
#import QTimer
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images
from image_processing import crop_image
import numpy as np


def combine_images(image1, image2):
    # 将图像转换为NumPy数组
    array1 = np.array(image1)
    array2 = np.array(image2)

    # 创建一个空白图像用于存储结果
    result_array = np.zeros_like(array1)

    # 找到两个图像中都是255的地方置为255，其他地方置为0
    mask = (array1 == 255) & (array2 == 255)
    result_array[mask] = 255
    print('ok')

    # 将结果数组转换回图像
    result_image = Image.fromarray(result_array)
    return result_image

def run_segmentation(self, index):
    if index == 0:
        return

    selected_items = self.patient_list.selectedItems()
    if selected_items and self.current_image_path:
        image_path = self.current_image_path

        if os.path.exists(image_path):
            # 打开原图像
            original_image = Image.open(image_path).convert("RGBA")
            width, height = original_image.size
            
            # 获取分割结果图像
            if index == 1:
                if os.path.isfile(self.rongmao_path):
                    self.rongmao = Image.open(self.rongmao_path)
                else:
                    self.rongmao = crop_image(self.model, image_path, self.device,index).convert("L")
                    self.rongmao.save(self.rongmao_path)
                segmentation_result = self.rongmao
                base_name, _ = os.path.splitext(os.path.basename(self.rongmao_path))
            elif index == 2:
                if os.path.isfile(self.shuizhong_path):
                    self.shuizhong = Image.open(self.shuizhong_path)
                else:
                    self.shuizhong = crop_image(self.shuimodel, image_path, self.device,index).convert("L")
                    if os.path.isfile(self.rongmao_path):
                        self.rongmao = Image.open(self.rongmao_path)
                    else:
                        self.rongmao = crop_image(self.model, image_path, self.device,index).convert("L")
                        self.rongmao.save(self.rongmao_path)
                    self.shuizhong = combine_images(self.rongmao,self.shuizhong)
                    
                    self.shuizhong.save(self.shuizhong_path)
                segmentation_result = self.shuizhong
                base_name, _ = os.path.splitext(os.path.basename(self.shuizhong_path))
            elif index == 3:
                if os.path.isfile(self.zengsheng_path):
                    self.zengsheng = Image.open(self.zengsheng_path)
                else:
                    self.zengsheng = crop_image(self.zengmodel, image_path, self.device,index).convert("L")
                    self.zengsheng.save(self.zengsheng_path)
                segmentation_result = self.zengsheng
                base_name, _ = os.path.splitext(os.path.basename(self.zengsheng_path))

            new_file_name = f"{base_name}_combined.png"
            new_image_path = os.path.join(os.path.dirname(image_path), new_file_name)

            if os.path.isfile(new_image_path):
                print("history")
            else:
                # 创建红色透明叠加层
                overlay = Image.new("RGBA", original_image.size, (255, 0, 0, 0))
                overlay_data = overlay.load()
                segmentation_data = segmentation_result.load()
                for y in range(segmentation_result.height):
                    for x in range(segmentation_result.width):
                        if segmentation_data[x, y] == 255:  # 分类为1的部分
                            overlay_data[x, y] = (255, 0, 0, 128)  # 透明红色
                # 将叠加层与原图像合并
                combined_image = Image.alpha_composite(original_image, overlay)
                # 保存结果图像
                combined_image.save(new_image_path, format='PNG')


            self.image_bottom = QPixmap(new_image_path)
            self.image_item_bottom = QGraphicsPixmapItem(self.image_bottom)
            self.graphics_scene_bottom.clear()
            self.graphics_scene_bottom.addItem(self.image_item_bottom)
            self.graphics_scene_bottom.setSceneRect(QRectF(self.image_bottom.rect()).adjusted(-500, -500, 500, 500))
            self.zoom_image_bottom()