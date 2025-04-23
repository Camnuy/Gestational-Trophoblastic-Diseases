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



GPT4V_ENDPOINT =   # 确保端点正确
API_KEY = 

def send_patient_data_to_gpt(current_patient_data, diagnosis_report, api_key):
    # 设置请求头
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    # 构建请求的负载
    payload = {
        "model": "deepseek-chat",  # Ensure the model name is correct
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a specialized and experienced pathologist with expertise in gestational trophoblastic diseases.\n"
                    "Hydatidiform moles generally include complete hydatidiform mole, partial hydatidiform mole, and invasive hydatidiform mole. Their differences in histopathological sections are mainly reflected in villous structure, degree of edema, trophoblastic proliferation, and distribution characteristics. The main histological features of these three types are as follows:\n"
                    "Complete Hydatidiform Mole\n"
                    "Villous structure: Markedly hydropic villi forming variably sized cystic structures with intracytoplasmic fluid accumulation. The central portion typically shows prominent cistern-like structures.\n"
                    "Trophoblastic proliferation: Diffuse hyperplasia of trophoblasts (including cytotrophoblasts and syncytiotrophoblasts). The hyperplastic trophoblasts usually surround the villi.\n"
                    "Vascularization: Villous stroma typically lacks blood vessels.\n"
                    "Partial Hydatidiform Mole\n"
                    "Villous structure: Mixture of normal and hydropic villi. Hydropic villi have irregular, scalloped contours, with some showing central cisterns.\n"
                    "Trophoblastic proliferation: Focal and limited trophoblastic hyperplasia. Some hyperplastic trophoblasts radiate outward from the villous surface.\n"
                    "Vascularization: Blood vessels are visible in some villous stroma, and embryonic tissue or fetal blood vessels may occasionally be observed.\n"
                    "Invasive Hydatidiform Mole\n"
                    "Villous structure: Hydropic villi invade the myometrium or deeper tissues. Some invasive villi may involve extrauterine structures such as the vagina, vulva, broad ligament, or pelvis.\n"
                    "Trophoblastic proliferation: Abnormally hyperplastic trophoblasts infiltrate the myometrium and blood vessels, showing high atypia.\n"
                    "Vascularization: Prominent vascular invasion is visible around the villi.\n"
                    "Summary\n"
                    "Complete hydatidiform mole: Characterized by markedly hydropic villi and diffuse trophoblastic hyperplasia, with absent vasculature.\n"
                    "Partial hydatidiform mole: Shows mixed normal and hydropic villi with focal trophoblastic hyperplasia and visible blood vessels.\n"
                    "Invasive hydatidiform mole: Features myometrial invasion by hydropic villi accompanied by highly atypical trophoblastic proliferation and vascular invasion."
                )
            },
            {
                "role": "user",
                "content": (
                    "Patient information is as follows:\n"
                    f"Gender: {current_patient_data['gender']}\n"
                    f"Age: {current_patient_data['age']}\n"
                    f"Referring department: {current_patient_data['department']}\n"
                    f"Pathological section statistics show: {diagnosis_report}. Based on logical decision-making, the preliminary clinical diagnosis is {current_patient_data['result']}.\n\n"
                    "Please comprehensively analyze this patient's condition by integrating the above information, provide a symptom description and summary, and offer further examination and treatment recommendations. "
                    "If a hydatidiform mole is suspected, predict the specific type based on the statistical results and provide your reasoning. "
                    "Respond with a single unstructured paragraph not exceeding 300 words."
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