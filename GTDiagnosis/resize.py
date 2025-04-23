import os
from PIL import Image

# 设置 Pillow 的最大图像大小限制
Image.MAX_IMAGE_PIXELS = None

def resize_and_convert_images(input_folder, scale_factor=0.125):
    # 遍历每个患者文件夹
    for patient_folder in os.listdir(input_folder):
        patient_path = os.path.join(input_folder, patient_folder)
        if os.path.isdir(patient_path):
            # 遍历每个患者文件夹中的tif文件
            for image_file in os.listdir(patient_path):
                if image_file.lower().endswith('.tif'):
                    image_path = os.path.join(patient_path, image_file)
                    
                    try:
                        # 打开图像并调整分辨率
                        with Image.open(image_path) as img:
                            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # 构造新文件名
                            base_name, _ = os.path.splitext(image_file)
                            new_file_name = f"{base_name}_resized.png"
                            new_file_path = os.path.join(patient_path, new_file_name)
                            
                            # 保存为PNG格式
                            resized_img.save(new_file_path, format='PNG')
                            print(f"Saved resized image: {new_file_path}")
                    
                    except Image.DecompressionBombError:
                        print(f"Skipping {image_path} due to decompression bomb error")

# 调用函数，传入data文件夹路径
resize_and_convert_images('data', scale_factor=0.125)
