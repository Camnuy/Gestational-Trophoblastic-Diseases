import torch
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QRectF
import cv2
from PIL import Image, ImageFile
from torchvision import transforms
import torch.nn.functional as F
#import QTimer
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images



def process_video_frame(self):
    ret, frame = self.cap.read()
    
    if not ret:
        self.timer.stop()
        return
    frame = frame[self.del_border:-self.del_border, self.del_border:-self.del_border]
    height, width, _ = frame.shape
    side_length = min(height, width)
    start_x = (width - side_length) // 2
    start_y = (height - side_length) // 2
    frame = frame[start_y:start_y + side_length, start_x:start_x + side_length]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = self.orb.detectAndCompute(gray, None)

    if self.previous_image is not None:
        if self.previous_descriptors is not None and descriptors is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.previous_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) >= 4:
                src_pts = np.float32([self.previous_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                self.transform_width += H[0, 2]
                self.transform_height += H[1, 2]
            
                if self.stitched_image is None:
                    self.stitched_image = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH, 3), dtype=np.uint8)
                    self.stitched_image_without_label = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH, 3), dtype=np.uint8)
                    self.stitched_segmentation_zengsheng = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH), dtype=np.uint8)
                    self.stitched_segmentation_shuizhong = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH), dtype=np.uint8)
                    self.stitched_segmentation_rongmao = np.zeros((self.MAX_HEIGHT, self.MAX_WIDTH), dtype=np.uint8)
                    center_x = (self.MAX_WIDTH - self.previous_image.shape[1]) // 2
                    center_y = (self.MAX_HEIGHT - self.previous_image.shape[0]) // 2
                    self.stitched_image[center_y:center_y + self.previous_image.shape[0], center_x:center_x + self.previous_image.shape[1]] = self.previous_frame
                    self.stitched_image_without_label[center_y:center_y + self.previous_image.shape[0], center_x:center_x + self.previous_image.shape[1]] = self.previous_frame
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
                self.stitched_image_without_label[start_y:end_y, start_x:end_x] = frame
                
                segmentation_output_zengsheng, segmentation_output_shuizhong, segmentation_output_rongmao = self.segment_frame(frame)
                self.stitched_image[start_y:end_y, start_x:end_x] = self.apply_segmentation_overlay(self.stitched_image[start_y:end_y, start_x:end_x], segmentation_output_zengsheng, segmentation_output_shuizhong)
                
                # Add segmentation results to stitched images
                self.stitched_segmentation_zengsheng[start_y:end_y, start_x:end_x] = np.where(segmentation_output_zengsheng > 0.5, 255, 0)
                self.stitched_segmentation_shuizhong[start_y:end_y, start_x:end_x] = np.where(segmentation_output_shuizhong > 0.5, 255, 0)
                self.stitched_segmentation_rongmao[start_y:end_y, start_x:end_x] = np.where(segmentation_output_rongmao > 0.5, 255, 0)
                
    segmentation_output_zengsheng, segmentation_output_shuizhong, segmentation_output_rongmao = self.segment_frame(frame)
    frame = self.apply_segmentation_overlay(frame, segmentation_output_zengsheng, segmentation_output_shuizhong)
    
    self.previous_frame = frame
    self.previous_image = gray
    self.previous_keypoints = keypoints
    self.previous_descriptors = descriptors

    self.update_frame_display(self.graphics_view_top, frame)
    if self.stitched_image is not None:
        if self.whether_label == 1:
            self.update_frame_display(self.graphics_view_bottom, self.stitched_image)
        if self.whether_label == 0:
            self.update_frame_display(self.graphics_view_bottom, self.stitched_image_without_label)

def save_and_crop_images(self, stitched_image, zengsheng, shuizhong, rongmao):
    # Remove black borders from the stitched image and get the cropping coordinates
    self.crop_to_content_with_coords(stitched_image)
    stitched_image_cropped = stitched_image[self.y:self.y+self.h, self.x:self.x+self.w]
    
    # Use the same coordinates to crop the segmentation images
    zengsheng_cropped = zengsheng[self.y:self.y+self.h, self.x:self.x+self.w]
    shuizhong_cropped = shuizhong[self.y:self.y+self.h, self.x:self.x+self.w]
    rongmao_cropped = rongmao[self.y:self.y+self.h, self.x:self.x+self.w]

    # Save images
    cv2.imwrite('stitched_image.png', stitched_image_cropped)
    cv2.imwrite('segmentation_output_zengsheng.png', zengsheng_cropped)
    cv2.imwrite('segmentation_output_shuizhong.png', shuizhong_cropped)
    cv2.imwrite('segmentation_output_rongmao.png', rongmao_cropped)

def crop_to_content_with_coords(self, img):
    if len(img.shape) == 3:  # color image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img, 0, 0, img.shape[1], img.shape[0]  # 返回原图像和默认坐标
    self.x, self.y, self.w, self.h = cv2.boundingRect(coords)

def segment_frame(self, frame):
    original_frame = frame
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (512,512))
    
    input_frame = cv2.resize(input_frame, (518,518))  # 根据模型输入尺寸调整
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    input_frame = np.array(input_frame)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_frame = (input_frame - mean) / std
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_frame = transform(input_frame).unsqueeze(0).to(device).float()
    frame = transform(frame).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        output_zengsheng = self.zengmodel(input_frame)
        output_shuizhong = self.shuimodel(input_frame)
        output_rongmao = self.model(frame)
    output_zengsheng = F.softmax(output_zengsheng, dim=1)
    output_shuizhong = F.softmax(output_shuizhong, dim=1)
    output_rongmao = F.softmax(output_rongmao, dim=1)
    
    output_zengsheng = output_zengsheng.cpu().numpy()[0, 1, :, :]  # 获取类别为1的概率
    output_zengsheng = cv2.resize(output_zengsheng, (original_frame.shape[1], original_frame.shape[0]))  # 调整回原始尺寸
    output_shuizhong = output_shuizhong.cpu().numpy()[0, 1, :, :]
    output_shuizhong = cv2.resize(output_shuizhong, (original_frame.shape[1], original_frame.shape[0]))
    output_rongmao = output_rongmao.cpu().numpy()[0, 1, :, :]  # 获取类别为1的概率
    output_rongmao = cv2.resize(output_rongmao, (original_frame.shape[1], original_frame.shape[0]))  # 调整回原始尺寸
    return output_zengsheng, output_shuizhong, output_rongmao

def apply_segmentation_overlay(self, image, segmentation_output_zengsheng, segmentation_output_shuizhong):
    alpha = 0.5
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[segmentation_output_zengsheng > 0.5] = [0, 0, 255]  # 红色标记
    mask = segmentation_output_zengsheng > 0.5
    image[mask] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask]
    overlay[segmentation_output_shuizhong > 0.5] = [0, 255, 0]  # 绿色标记
    mask = segmentation_output_shuizhong > 0.5
    image[mask] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask]
    return image

def toggle_segmentation(self):
    if self.toggle_segmentation_button.isChecked():
        print("显微镜开始分割")
        self.whether_label = 1
    else:
        print("显微镜结束分割")
        self.whether_label = 0

def clear_right_image(self):
    # 清空拼接的图像
    self.stitched_image = None
    self.stitched_image_without_label = None
    self.stitched_segmentation_zengsheng = None
    self.stitched_segmentation_shuizhong = None
    self.stitched_segmentation_rongmao = None
    self.transform_width = 0
    self.transform_height = 0

    # 清空显示区域
    self.graphics_view_bottom.scene.clear()
    self.graphics_view_bottom.setSceneRect(QRectF())  # 重置场景矩形

    print("底部图片已清空并重新初始化，准备继续拼接")

def update_frame_display(self, scene, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    scene.scene.clear()
    scene.scene.addPixmap(pixmap)
    scene.setSceneRect(QRectF(pixmap.rect()))  # 设置场景矩形以使拖动有效