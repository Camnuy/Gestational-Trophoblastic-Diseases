import os
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable loading of truncated images
from image_processing import crop_image
from model_language import *

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

def calculate_metrics_with_count(self):
    # Convert images to numpy arrays
    rongmao_array = np.array(self.rongmao)
    shuizhong_array = np.array(self.shuizhong)
    zengsheng_array = np.array(self.zengsheng)

    # Calculate total pixels
    total_pixels = rongmao_array.size

    # Calculate pixel areas
    villi_pixels = np.sum(rongmao_array == 255)
    edema_pixels = np.sum(shuizhong_array == 255)
    hyperplasia_pixels = np.sum(zengsheng_array == 255)

    # Calculate area ratios
    villi_ratio = villi_pixels / total_pixels
    edema_ratio = edema_pixels / villi_pixels if villi_pixels > 0 else 0
    hyperplasia_ratio = hyperplasia_pixels / villi_pixels if villi_pixels > 0 else 0
    abnormal_villi_ratio = (edema_pixels + hyperplasia_pixels) / villi_pixels if villi_pixels > 0 else 0

    # Preprocessing for connected component analysis
    kernel = np.ones((3,3), np.uint8)

    # Erosion to remove noise and small objects
    villi_eroded = cv2.erode(rongmao_array, kernel, iterations=1)
    edema_eroded = cv2.erode(shuizhong_array, kernel, iterations=1)
    hyperplasia_eroded = cv2.erode(zengsheng_array, kernel, iterations=1)

    # Connected component analysis
    _, villi_labels, villi_stats, _ = cv2.connectedComponentsWithStats(villi_eroded, connectivity=8)
    _, edema_labels, edema_stats, _ = cv2.connectedComponentsWithStats(edema_eroded, connectivity=8)
    _, hyperplasia_labels, hyperplasia_stats, _ = cv2.connectedComponentsWithStats(hyperplasia_eroded, connectivity=8)

    # Exclude the background label
    villi_count = len(villi_stats) - 1
    edema_count = len(edema_stats) - 1
    hyperplasia_count = len(hyperplasia_stats) - 1

    # Calculate count ratios
    total_count = villi_count
    edema_count_ratio = edema_count / total_count if total_count > 0 else 0
    hyperplasia_count_ratio = hyperplasia_count / total_count if total_count > 0 else 0
    abnormal_villi_count_ratio = (edema_count + hyperplasia_count) / total_count if total_count > 0 else 0

    return {
        "total_pixels": total_pixels,
        "villi_pixels": villi_pixels,
        "edema_pixels": edema_pixels,
        "hyperplasia_pixels": hyperplasia_pixels,
        "villi_ratio": villi_ratio,
        "edema_ratio": edema_ratio,
        "hyperplasia_ratio": hyperplasia_ratio,
        "abnormal_villi_ratio": abnormal_villi_ratio,
        "total_count": total_count,
        "villi_count": villi_count,
        "edema_count": edema_count,
        "hyperplasia_count": hyperplasia_count,
        "edema_count_ratio": edema_count_ratio,
        "hyperplasia_count_ratio": hyperplasia_count_ratio,
        "abnormal_villi_count_ratio": abnormal_villi_count_ratio
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
            return "视野内见胎盘绒毛及蜕膜组织，部分绒毛水肿明显，滋养层细胞可见增生，考虑葡萄胎，建议进行分子STR检测进一步明确诊断。"
        elif metrics["edema_ratio"] > edema_threshold2 and metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
            return "视野内见胎盘绒毛及蜕膜组织，部分绒毛水肿明显，滋养层细胞局部增生，考虑葡萄胎，建议进行分子STR检测进一步明确诊断。"
        elif metrics["edema_ratio"] > edema_threshold1 and metrics["hyperplasia_ratio"] > hyperplasia_threshold2:
            return "视野内见胎盘绒毛及蜕膜组织，部分绒毛水肿，滋养层细胞局部增生，考虑葡萄胎，建议进行分子STR检测进一步明确诊断。"
        
        elif metrics["edema_ratio"] > edema_threshold1 and metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
            return"视野内见胎盘绒毛及蜕膜组织，绒毛存在轻度水肿，局灶滋养细胞增生，考虑为整体轻度异常，建议监测血HCG。"
        elif metrics["edema_ratio"] > edema_threshold1:
            return"视野内见胎盘绒毛及蜕膜组织，绒毛存在轻度水肿，考虑为整体轻度异常，建议监测血HCG。"
        elif metrics["hyperplasia_ratio"] > hyperplasia_threshold1:
            return"视野内见胎盘绒毛及蜕膜组织，局灶滋养细胞增生，考虑为整体轻度异常，建议监测血HCG。"
        
        return "胎盘绒毛及蜕膜组织及部分子宫内膜，绒毛形态基本正常，考虑为正常流产"

def generate_report(self, metrics, classification):
    report = f"统计报告:\n\n"
    report += f"绒毛组织面积占比: {metrics['villi_pixels']}pixels ({metrics['villi_ratio']*100:.2f}%)\n"
    report += f"水肿组织面积占比: {metrics['edema_pixels']}pixels ({metrics['edema_ratio']*100:.2f}%)\n"
    report += f"增生组织面积占比: {metrics['hyperplasia_pixels']}pixels ({metrics['hyperplasia_ratio']*100:.2f}%)\n"
    report += f"绒毛组织个数: {metrics['villi_count']}\n"
    report += f"水肿组织个数: {metrics['edema_count']} ({metrics['edema_count_ratio']*100:.2f}%)\n"
    report += f"增生组织个数: {metrics['hyperplasia_count']} ({metrics['hyperplasia_count_ratio']*100:.2f}%)\n"
    report += f"病理诊断:\n  {classification}"

    diagnosis_report = f"病理诊断:\n  {classification}"

    fig = self.plot_pie_charts(metrics)
    canvas = FigureCanvas(fig)
    canvas_layout = QVBoxLayout(self.canvas_widget)
    canvas_layout.addWidget(canvas)
    self.canvas_widget.setLayout(canvas_layout)
    self.right_widget.setFixedWidth(750) # Set the height to show the chart

    return report, diagnosis_report

def generate_pie_chart(self, metrics):
    area_labels = ['正常绒毛', '水肿绒毛']
    villi_area = metrics['villi_pixels']
    edema_area = metrics['edema_pixels']
    normal_area = villi_area - edema_area
    areas = [max(normal_area, 0), max(edema_area,0)]
    count_labels = ['正常绒毛', '水肿绒毛', '增生绒毛', '水肿增生']
    villi_count = metrics['villi_count']
    edema_count = metrics['edema_count']
    hyperplasia_count = metrics['hyperplasia_count']
    if edema_count + hyperplasia_count > villi_count:
        overlap_count = edema_count + hyperplasia_count - villi_count
    else:
        overlap_count = 0
    normal_count = villi_count - edema_count - hyperplasia_count + overlap_count
    counts = [max(normal_count, 0), max(edema_count - overlap_count, 0), max(hyperplasia_count - overlap_count, 0), max(overlap_count, 0)]
    print(areas)
    print(counts)

    return area_labels, areas, count_labels, counts

def plot_pie_charts(self ,metrics):
    area_labels, areas, count_labels, counts = self.generate_pie_chart(metrics)
    font_properties_title = FontProperties(fname='res/simhei.ttf', size=12)  # 修改为你系统中的中文字体路径
    font_properties_legend = FontProperties(fname='res/simhei.ttf', size=8)  # 修改为你系统中的中文字体路径

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 12))
    ax1.set_facecolor('#FFFFFF')
    colors = ['lightgreen', 'pink', 'gold', 'darkred']
    # Plot area pie chart
    wedges1, texts1, autotexts1 = ax1.pie(areas, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 8}, wedgeprops={'width': 1, 'edgecolor': 'w'})
    ax1.set_title('绒毛面积分布', fontproperties=font_properties_title, pad=0, y=0.93)
    ax1.legend(wedges1, area_labels, loc='upper center', bbox_to_anchor=(1, 0.5), prop=font_properties_legend)
    # 移除饼状图内的标签文字
    for text in texts1:
        text.set_text('')

    # Plot count pie chart
    wedges2, texts2, autotexts2 = ax2.pie(counts, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 8}, wedgeprops={'width': 0.5, 'edgecolor': 'w'})
    ax2.set_title('绒毛个数分布', fontproperties=font_properties_title, pad=0, y=0.93)
    ax2.legend(wedges2, count_labels, loc='upper center', bbox_to_anchor=(1, 0.5), prop=font_properties_legend)
    ax2.set_facecolor('#FFFFFF')
    for text in texts2:
        text.set_text('')
    # 调整饼状图之间的间距和周围空白间隔
    plt.subplots_adjust(hspace=-0.2, top=1, bottom=0)
    plt.tight_layout(pad=0)  # 全局调整布局填充
    fig.patch.set_facecolor('#F0F4F8')  # 设置整个图的背景颜色
    return fig

def diagnosis(self):
    if self.mode == 1:
        if os.path.isfile(self.rongmao_path) and os.path.isfile(self.shuizhong_path) and os.path.isfile(self.zengsheng_path):
            self.rongmao = Image.open(self.rongmao_path[:-4]+'_diagnosis.png')
            self.shuizhong = Image.open(self.shuizhong_path)
            self.zengsheng = Image.open(self.zengsheng_path)
        else:
            self.run_segmentation_online(index=1)
            self.rongmao = Image.open(self.rongmao_path[:-4]+'_diagnosis.png')
            self.shuizhong = Image.open(self.shuizhong_path)
            self.zengsheng = Image.open(self.zengsheng_path)
    if self.mode == 2:
        self.save_and_crop_images(self.stitched_image, self.stitched_segmentation_zengsheng, self.stitched_segmentation_shuizhong, self.stitched_segmentation_rongmao)
        self.rongmao = Image.open('segmentation_output_rongmao.png')
        self.shuizhong = Image.open('segmentation_output_shuizhong.png')
        self.zengsheng = Image.open('segmentation_output_zengsheng.png')
    
    metrics = self.calculate_metrics_with_count()
    classification = self.classify_slide(metrics)
    report,diagnosis_report = self.generate_report(metrics, classification)
    print(diagnosis_report)
    print(self.current_patient_data)
    # diagnosis_response_new = send_patient_data_to_gpt(self.current_patient_data, diagnosis_report, API_KEY)
    # print("deepseek Response:", diagnosis_response_new)

    self.detailed_info_text.setText(report)
    
    # self.report_text.setText(f"deepseek辅助诊断:\n  {diagnosis_response_new}")
    self.report_text.setText(f"病理诊断:\n  {diagnosis_report}")
    self.pdf_text = diagnosis_report