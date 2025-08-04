import re
import pandas as pd

# Function to normalize numbering in answer choices only and log altered texts
def normalize_choices(text):
    numbering_map = {
        '1)': '1)', '1.': '1)', '(1)': '1)', '①': '1)', 'A)': '1)', '가)': '1)', '가.': '1)',
        '2)': '2)', '2.': '2)', '(2)': '2)', '②': '2)', 'B)': '2)', '나)': '2)', '나.': '2)',
        '3)': '3)', '3.': '3)', '(3)': '3)', '③': '3)', 'C)': '3)', '다)': '3)', '다.': '3)',
        '4)': '4)', '4.': '4)', '(4)': '4)', '④': '4)', 'D)': '4)', '라)': '4)', '라.': '4)',
        '5)': '5)', '5.': '5)', '(5)': '5)', '⑤': '5)', 'E)': '5)', '마)': '5)', '마.': '5)'
    }

    # Split the text by newlines
    parts = text.split('\n')
    altered = False  # Track if any alteration is made
    
    # Iterate over each part, checking if it starts with a numbering pattern
    for i in range(1, len(parts)):  # Start from 1 to skip question part
        # Match only if the part starts with a numbering pattern
        match = re.match(r"^(\d+\)|\d+\.|\(\d+\)|①|②|③|④|⑤|[A-E]\)|[가-마]+\)|[가-마]+\.)", parts[i])
        if match:
            original_numbering = match.group(0)
            # Replace with the correct numbering if found in the mapping
            normalized_numbering = numbering_map.get(original_numbering, original_numbering)
            if original_numbering != normalized_numbering:  # Check if replacement is needed
                parts[i] = parts[i].replace(original_numbering, normalized_numbering, 1)
                altered = True  # Mark as altered
    
    # Join the parts back together
    normalized_text = '\n'.join(parts)
    
    # # Print the original and altered text if any changes were made
    # if altered:
    #     print("Original Text:\n", text)
    #     print("Altered Text:\n", normalized_text)
    #     print("-" * 50)
    
    return normalized_text

def normalize_numbering(text):
    numbering_map = {
        '1)': '1)', '1.': '1)', '(1)': '1)', '①': '1)', 'A)': '1)', '가)': '1)', '가.': '1)',
        '2)': '2)', '2.': '2)', '(2)': '2)', '②': '2)', 'B)': '2)', '나)': '2)', '나.': '2)',
        '3)': '3)', '3.': '3)', '(3)': '3)', '③': '3)', 'C)': '3)', '다)': '3)', '다.': '3)',
        '4)': '4)', '4.': '4)', '(4)': '4)', '④': '4)', 'D)': '4)', '라)': '4)', '라.': '4)',
        '5)': '5)', '5.': '5)', '(5)': '5)', '⑤': '5)', 'E)': '5)', '마)': '5)', '마.': '5)'
    }
    # Find and replace each numbering pattern at the beginning of the text
    pattern = r"^(\d+\)|\d+\.|\(\d+\)|①|②|③|④|⑤|[A-E]\)|[가-마]+\)|[가-마]+\.)"
    match = re.match(pattern, text)
    if match:
        original_numbering = match.group(0)
        normalized_numbering = numbering_map.get(original_numbering, original_numbering)
        return text.replace(original_numbering, normalized_numbering, 1)
    return text

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = train_data[train_data['q_type'] == 1]
    test_data = test_data.rename(columns={"user": "question"})
    return train_data, test_data        

def load_full_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)    
    test_data = test_data.rename(columns={"user": "question"})
    return train_data, test_data


def generate_prompt(example):
    question = example["question"]
    answer = example.get("answer", "")
    prompt = f"""다음 객관식 문제를 읽고 5개의 보기 중 1개의 정답을 고르시오.
문제: {question}
답변: {answer}""".strip()
    return prompt

def generate_prompt_multiple(example):
    question = example["question"]
    answer = example.get("answer", "")
    prompt = f"""다음 객관식 문제를 읽고 가장 적절한 보기 하나를 선택하시오.
문제: {question}
답변: {answer}""".strip()
    return prompt

def generate_prompt_descriptive(example):
    question = example["question"]
    answer = example.get("answer", "")
    prompt = f"""다음의 질문에 대하여 간략한 답변을 작성하시오.
질문: {question}
답변: {answer}""".strip()
    return prompt

def generate_prompt_with_cot(example):
    question = example["question"]
    cot = example["cot"]
    answer = example.get("answer", "")
    prompt = f"""
문제: {question}
추론: {cot}
실제 정답: {answer}""".strip()
    return prompt

# def extract_answer(text):
#     # Match the answer choice right after "답변:" or at the beginning of labels
#     match = re.search(r"(?:답변:\s*|^)(\d+)\)?", text)
#     if match:
#         return match.group(1)  # Extracts the choice number (e.g., "1" from "1)" or "답변: 1")
#     return None
def extract_answer(text):
    # Match the answer choice right after "답변:" or at the beginning of labels
    match = re.search(r"(?:정답:\s*|^)(\d+)\)?", text)
    if match:
        return match.group(1)  # Extracts the choice number (e.g., "1" from "1)" or "답변: 1")
    return None

def extract_answer_instruct(text):    
    match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(\d+)\)", text)
    if match:
        return match.group(1)  # Extracts the choice number (e.g., "3" from "3)")
    return None

def extract_answer_qwen_cot(text):
    # Match the pattern "답: <number>)"
    match = re.search(r"답:\s*(\d+)\)", text)
    if match:
        return match.group(1)  # Extract the number part
    return None

# def extract_answer_instruct_qwen(text):
#     # Match the answer number followed by <|im_start|>assistant
#     match = re.search(r"<\|im_start\|>assistant\s*(\d+)\)", text)
#     if match:
#         return match.group(1)  # Extracts the choice number (e.g., "3" from "3)")
#     return None

def extract_answer_instruct_qwen(text):
    # Match the answer number followed by <|im_start|>assistant
    match = re.search(r"<\|im_start\|>assistant\s*(\d+)", text)
    if match:
        return match.group(1)  # Extracts the choice number (e.g., "3" from "3")
    return None

def extract_answer_num(text):
    match = re.match(r"^(\d+)\)", text)
    if match:
        first_number = match.group(1)  # Extract the first number
        return first_number
    return ""
