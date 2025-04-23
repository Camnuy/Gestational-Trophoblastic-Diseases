# import openai
# import requests
# import time
# import datetime
# import json

# # GPT4V_ENDPOINT = 'https://cvpr.openai.azure.com/openai/deployments/Lab_429_4/chat/completions?api-version=2024-02-15-preview'
# # API_KEY = '92601e0ae3824b51a45fa68e01d9c4c4'  # 请替换为您的实际API密钥
# GPT4V_ENDPOINT = 'https://api.deepseek.com'  #ds
# API_KEY = 'sk-79764df57f07469cb5c64559104c31c9'
# def send_patient_data_to_gpt(current_patient_data, diagnosis_report, api_key):
#     # 设置请求头
#     # headers = {
#     #     'Content-Type': 'application/json',
#     #     'api-key': api_key,
#     # }
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {api_key}',
#     }

#     # 构建请求的负载
#     payload = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     f"您是一位专业且经验丰富的病理学家，尤其是针对妊娠滋养细胞疾病。\n"
#                     "葡萄胎一般包括完全性葡萄胎、部分性葡萄胎和侵蚀性葡萄胎，其在组织学病理切片上的差异主要体现在绒毛结构、水肿程度、滋养细胞增生及其分布特征等方面。以下是三者的主要组织学特点："
#                     "完全性葡萄胎"
#                     "绒毛结构：绒毛高度水肿，呈现大小不等的囊泡样结构，内部液体积聚。中央部分通常有明显的水池样结构。"
#                     "滋养细胞增生：滋养细胞（包括细胞滋养细胞和合体滋养细胞）弥漫性增生。增生的滋养细胞通常环绕绒毛周围。"
#                     "血管化：绒毛间质通常缺乏血管。"
#                     "部分性葡萄胎"
#                     "绒毛结构：正常绒毛与水肿绒毛混合存在。水肿绒毛轮廓不规则，呈扇贝状，部分绒毛有中央水池。"
#                     "滋养细胞增生：滋养细胞增生局灶性，较为局限。部分增生的滋养细胞从绒毛表面向外放射状排列。"
#                     "血管化：部分绒毛间质内可见血管，有时可以观察到胚胎组织或胎儿血管。"
#                     "侵蚀性葡萄胎"
#                     "绒毛结构：水肿绒毛侵入子宫肌层或更深的部位。部分侵入的绒毛可以累及子宫外的结构，如阴道、外阴、阔韧带或盆腔。"
#                     "滋养细胞增生：异常增生的滋养细胞浸润子宫肌层和血管。增生的滋养细胞异型性较高。"
#                     "血管化：绒毛周围可见明显的血管侵袭。"
#                     "总结"
#                     "完全性葡萄胎：以高度水肿的绒毛和弥漫性滋养细胞增生为特征，无血管。"
#                     "部分性葡萄胎：存在正常和水肿绒毛混合，局灶性滋养细胞增生，可见血管。"
#                     "侵蚀性葡萄胎：水肿绒毛侵入子宫肌层，伴有高度异型增生的滋养细胞浸润和血管侵袭。"
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"患者信息如下：\n"
#                     f"性别: {current_patient_data['gender']}\n"
#                     f"年龄: {current_patient_data['age']}\n"
#                     f"送检科室: {current_patient_data['department']}\n"
#                     f"宫腔组织病理切片统计结果显示：{diagnosis_report}，根据逻辑决策，初步临床诊断为{current_patient_data['result']}。\n\n"
#                     "请综合以上信息，对这个患者进行症状描述、总结和分析，并进一步给出检查与治疗建议。"
#                     "如果可能是葡萄胎，请根据统计结果预测葡萄胎的种类，并给出理由。"
#                     "要求仅返回一段没有结构的文字，字数不超过300字。"
#                 )
#             }
#         ],
#         "temperature": 0.7,
#         "max_tokens": 500,
#         "top_p": 1,
#     }

#     def send_request_with_retry():
#         max_retries = 3
#         base_delay = 60  # seconds
#         backoff_factor = 2

#         for retry in range(max_retries):
#             try:
#                 start_time = time.time()
#                 response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
#                 response.raise_for_status()  # 如果返回状态码是失败的，会引发HTTPError
#                 json_response = response.json()
#                 break  # 如果请求成功，跳出循环
#             except requests.RequestException as e:
#                 if response is not None and response.status_code == 429:
#                     delay = base_delay * (backoff_factor ** retry)  # 指数退避
#                     print(f"Received 429 error. Retrying in {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     raise SystemExit(f"Failed to make the request. Error: {e}")

#         elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
#         print("--- Elapsed Time: %s ---" % elapsed_time)
#         print(json_response.get("usage", {}))

#         return json_response["choices"][0]["message"]["content"]

#     return send_request_with_retry()

import openai
import requests
import time
import datetime
import json
import openai
import requests
import time
import datetime
import json

diagnosis_report = "视野内见胎盘绒毛及蜕膜组织，绒毛存在轻度水肿，局灶滋养细胞增生，考虑为整体轻度异常，建议监测血HCG"
current_patient_data = {
'patient_index': 'H2400413', 'name': '李某', 'gender': '女', 'age': 23, 'department': '妇科', 'submission_time': '2024.07.08', 'admission_number': '12345', 'bed_number': '429', 'doctor': '阙文戈', 'result': '葡萄胎', 'diagnosis_report': '根据患者苍岳洋的临床诊断与病理切片分析，绒毛组织、水肿组织、增生 组织的统计显示异常绒毛面积和个数占比明显增高，这与葡萄胎的特征相符。葡萄胎是一种滋养细胞疾病，表现为绒毛水肿和变性，伴有绒毛表皮的过度增生。建议对患者进行全面的妇科检查和血液HCG水平监测，以评估疾病的进展。治疗上，通常采用清宫手术来移除异常组织，并在手术后密切监测HCG水平，以确保疾病彻底治愈并防止恶变。', 'folder': '0003'
}

GPT4V_ENDPOINT = 'https://api.deepseek.com/v1/chat/completions'  # 确保端点正确
API_KEY = 'sk-79764df57f07469cb5c64559104c31c9'

def send_patient_data_to_gpt(current_patient_data, diagnosis_report, api_key):
    # 设置请求头
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    # 构建请求的负载
    payload = {
        "model": "deepseek-chat",  # 确保模型名称正确
        "messages": [
            {
                "role": "system",
                "content": (
                    f"您是一位专业且经验丰富的病理学家，尤其是针对妊娠滋养细胞疾病。\n"
                    "葡萄胎一般包括完全性葡萄胎、部分性葡萄胎和侵蚀性葡萄胎，其在组织学病理切片上的差异主要体现在绒毛结构、水肿程度、滋养细胞增生及其分布特征等方面。以下是三者的主要组织学特点："
                    "完全性葡萄胎"
                    "绒毛结构：绒毛高度水肿，呈现大小不等的囊泡样结构，内部液体积聚。中央部分通常有明显的水池样结构。"
                    "滋养细胞增生：滋养细胞（包括细胞滋养细胞和合体滋养细胞）弥漫性增生。增生的滋养细胞通常环绕绒毛周围。"
                    "血管化：绒毛间质通常缺乏血管。"
                    "部分性葡萄胎"
                    "绒毛结构：正常绒毛与水肿绒毛混合存在。水肿绒毛轮廓不规则，呈扇贝状，部分绒毛有中央水池。"
                    "滋养细胞增生：滋养细胞增生局灶性，较为局限。部分增生的滋养细胞从绒毛表面向外放射状排列。"
                    "血管化：部分绒毛间质内可见血管，有时可以观察到胚胎组织或胎儿血管。"
                    "侵蚀性葡萄胎"
                    "绒毛结构：水肿绒毛侵入子宫肌层或更深的部位。部分侵入的绒毛可以累及子宫外的结构，如阴道、外阴、阔韧带或盆腔。"
                    "滋养细胞增生：异常增生的滋养细胞浸润子宫肌层和血管。增生的滋养细胞异型性较高。"
                    "血管化：绒毛周围可见明显的血管侵袭。"
                    "总结"
                    "完全性葡萄胎：以高度水肿的绒毛和弥漫性滋养细胞增生为特征，无血管。"
                    "部分性葡萄胎：存在正常和水肿绒毛混合，局灶性滋养细胞增生，可见血管。"
                    "侵蚀性葡萄胎：水肿绒毛侵入子宫肌层，伴有高度异型增生的滋养细胞浸润和血管侵袭。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"患者信息如下：\n"
                    f"性别: {current_patient_data['gender']}\n"
                    f"年龄: {current_patient_data['age']}\n"
                    f"送检科室: {current_patient_data['department']}\n"
                    f"宫腔组织病理切片统计结果显示：{diagnosis_report}，根据逻辑决策，初步临床诊断为{current_patient_data['result']}。\n\n"
                    "请综合以上信息，对这个患者进行症状描述、总结和分析，并进一步给出检查与治疗建议。"
                    "如果可能是葡萄胎，请根据统计结果预测葡萄胎的种类，并给出理由。"
                    "要求仅返回一段没有结构的文字，字数不超过300字。"
                )
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
    }

    def send_request_with_retry():
        max_retries = 3
        base_delay = 60  # seconds
        backoff_factor = 2

        for retry in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()  # 如果返回状态码是失败的，会引发HTTPError
                json_response = response.json()
                break  # 如果请求成功，跳出循环
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    delay = base_delay * (backoff_factor ** retry)  # 指数退避
                    print(f"Received 429 error. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise SystemExit(f"Failed to make the request. Error: {e}")
            except requests.exceptions.RequestException as e:
                raise SystemExit(f"Failed to make the request. Error: {e}")

        elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
        print("--- Elapsed Time: %s ---" % elapsed_time)
        print(json_response.get("usage", {}))

        return json_response["choices"][0]["message"]["content"]

    return send_request_with_retry()