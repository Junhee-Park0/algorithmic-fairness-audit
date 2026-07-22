import numpy as np
import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from typing import List
from prompts import background, data_context, system_prompt, user_prompt as USER_PROMPT_TEMPLATE
from config import RAW_DATA_PATH, OUTPUT_PATH
import dotenv
dotenv.load_dotenv()

raw_data_path = Path(RAW_DATA_PATH)
output_path = Path(OUTPUT_PATH)
os.makedirs(output_path, exist_ok = True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

def convert_numpy(obj):
    """JSON 직렬화 에러 방지용"""
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy(v) for v in obj]
    return obj

def get_target_files(target_variables: List[str] = None):
    """분석 대상 파일 선택 (None이면 폴더 내 모든 json 분석)"""
    all_files = glob.glob(f"{raw_data_path}/*.json")
    if not target_variables:
        return all_files
    target_files = [file for file in all_files if Path(file).stem in {variable for variable in target_variables}]
    return target_files

def generate_report(target_files: List[str], business_objective):
    """분석 대상 파일에 대한 보고서 생성"""
    if not target_files:
        print("모든 변수에 대해 진행합니다.")

    for file in tqdm(target_files, desc = "Generating Reports"):
        with open(file, "r", encoding = "utf-8") as f:
            final_report_json = convert_numpy(json.load(f))
            feature_name = Path(file).stem

            user_prompt = USER_PROMPT_TEMPLATE.format(
                background = background,
                data_context = data_context,
                feature_name = feature_name,
                business_objective = business_objective,
                final_report_json = json.dumps(final_report_json, ensure_ascii = False, indent = 2)
            )

            try:
                response = client.chat.completions.create(
                    model = "gpt-4o",
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature = 0.15
                )

                report_text = response.choices[0].message.content

                output_file = os.path.join(output_path, f"Audit_Report_{feature_name}.md")
                with open(output_file, "w", encoding = "utf-8") as out_f:
                    out_f.write(report_text)
                print(f"✅ {feature_name} 분석 보고서 작성 완료")
                
            except Exception as e:
                print(f"❌ Error processing {feature_name}: {e}")


if __name__ == "__main__":
    target_vars = ["Age & Gender"]
    target_files = get_target_files(target_vars)
    business_objective = "해지하지 않을 고객을 대상으로 마케팅을 할 건데, 마케팅 비용이 비싸. 최대한 진짜로 남을 사람들만 유지로 판단해줘"
    generate_report(target_files, business_objective)
