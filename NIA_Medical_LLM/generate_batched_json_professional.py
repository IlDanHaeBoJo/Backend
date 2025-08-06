import pandas as pd
import json

# Load and sample data
professional_csv = pd.read_csv("/workspace/datasets/241030_[훈련용] 전문QA_ver01 (14753).csv")
#sampled_data = essential_csv.sample(10)  # Adjust the sample size as needed

# Prepare batch requests
requests = []
for index, row in professional_csv.iterrows():
    question = row['question']
    answer = row['answer'].strip()

    # Define the prompt
    prompt = f"""당신은 전문 의학 지식을 가진 의사입니다. 주어진 문제에 대해 논리적인 사고 과정을 통해 정답을 도출하세요. 각 단계를 명확히 나누어 작성하고 마지막에 정답을 선택하세요.
Format:
- 문제 분석: ...
- 가능성 있는 답안 분석: ...
- 정답 결정 이유: ...
답: 1) [정답] \n
"""

    # Create the request
    request_body = {
        "custom_id": f"request-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"문제: {question}"}
            ],
            "max_tokens": 600
        }
    }
    requests.append(request_body)

# Save requests to a JSONL file
batch_file_path = "/workspace/datasets/batch_requests_professional.jsonl"
with open(batch_file_path, "w") as batch_file:
    for request in requests:
        batch_file.write(json.dumps(request) + "\n")

print(f"Batch requests saved to {batch_file_path}")

# Simulate sending requests (depends on your API environment)
# Here you would read the JSONL file and send requests in bulk using an API client or tool that supports batch processing.
