import torch
import numpy as np
import cv2
import math
from torchvision import transforms
from PIL import Image, ImageFile
import os
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

def crop_image(model, image_path, device, index,patch_size=512, step_size=None,):
    Image.MAX_IMAGE_PIXELS = 4294836225  # 防止出现图片太大无法读取
    image = Image.open(image_path)
    width, height = image.size
    pre_label = np.zeros((height, width), dtype=np.uint8)
    if step_size is None:
        step_size = patch_size
    width_num = math.ceil(width / step_size)
    height_num = math.ceil(height / step_size)
    total_iterations = width_num * height_num
    dir_path = os.path.dirname(image_path)
    
    for i in range(0, width_num):
        for j in range(0, height_num):
            current_iteration = i * height_num + j + 1
            # 计算进度百分比
            progress_percent = (current_iteration / total_iterations) * 100
            # 输出进度百分比
            print(f"Processing: {progress_percent:.2f}% completed")
            width_start = min(step_size * i, width - patch_size)
            height_start = min(step_size * j, height - patch_size)
            width_end = width_start + patch_size
            height_end = height_start + patch_size
            '''
            crop = image.crop((width_start, height_start, width_end, height_end))  # 修正此行
            crop.save(dir_path+'/crops/'+str(i)+'_'+str(j)+'.png')
            '''
            crop = Image.open(dir_path+'/crops/'+str(i)+'_'+str(j)+'.png')
            variance = np.var(np.array(crop)) 
            if variance > 0:
                pre_crop = predict(model=model, img=crop, device=device,index = index)
                pre_crop_img = Image.fromarray(pre_crop) 
                pre_crop_img = pre_crop_img.resize((patch_size, patch_size), Image.NEAREST)
                pre_crop = np.array(pre_crop_img) 
                pre_label[height_start:height_end, width_start:width_end] = pre_crop
            else:
                pre_label[height_start:height_end, width_start:width_end] = 0
    
    pre_label = Image.fromarray(pre_label)
    pre_label = np.array(pre_label)
    pre_label = remove_small_holes(pre_label)
    pre_label = smooth_edges(pre_label)
    pre_label_img = Image.fromarray(pre_label)
    return pre_label_img
     


def predict(model, img, device ,index):
    if(index==1):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = np.array(img)
        img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = (img - mean) / std
    img = transform(img).unsqueeze(0).to(device).float()

    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).cpu().squeeze().numpy()
    pred = np.where(pred == 1, 255, 0).astype(np.uint8)
    pred = pred[3:-3, 3:-3]
    return pred