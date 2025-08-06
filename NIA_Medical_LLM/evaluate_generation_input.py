import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import normalize_choices, normalize_numbering, load_data, generate_prompt, extract_answer, extract_answer_num
from tqdm import tqdm
import logging
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="evaluation_of_finetuned_qwen_model")
    parser.add_argument("--model_name_or_path", default="./results/qwen_14b_professional_lora_cot/checkpoint-56000", type=str)
    parser.add_argument("--output_path", default="./results/qwen_14b_evaluation_results_professional_lora_cot-56000_test.csv", type=str)
    parser.add_argument("--essential_data_path", default="./datasets/241030_[훈련용] 필수QA_ver01 (18627).csv", type=str)
    parser.add_argument("--professional_data_path", default="./datasets/241030_[훈련용] 전문QA_ver01 (14753).csv", type=str)
    parser.add_argument("--test_data_path", default="./datasets/241030_[테스트용] KorMedMCQA_ver01 (2494).csv", type=str)
    parser.add_argument("--generation_data_path", default=None, type=str)
    parser.add_argument("--generation_data_question_column_name", default='question', type=str)
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--log_file", default="test_execution_log.txt", type=str)

    return parser.parse_args()

def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    # Log start
    logging.info("Starting evaluation script.")
    logging.info(f"Arguments: {args}")

    logging.info(f"mode: {args.mode}")

    if torch.cuda.is_available():
        logging.info(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device_map_setting = "auto"
    else:
        logging.warning("CUDA is NOT available. The model will run on CPU, which might be very slow.")
        device_map_setting = "cpu" # 명시적으로 CPU 사용을 설정

    # Load model and tokenizer
    model_name = args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map_setting, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logging.info(f"Model and tokenizer loaded from {model_name}")
    max_new_tokens = 512

    if args.mode == "professional":
        data = pd.read_csv(args.professional_data_path)        
    elif args.mode == "essential":
        pass
    elif args.mode == "generation":
        # Define text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map=device_map_setting,
        )

        logging.info(f"Text generation pipeline created")

        def add_default_prompt(question, answer=""):
            prompt = f"""다음의 질문에 대하여 간략한 답변을 작성하시오.\n질문: {question}\n답변: {answer}""".strip()
            return prompt

        def generate_output(prompt, max_new_tokens=512, do_sample=False, no_repeat_ngram_size=None, num_beams=None):
            outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, no_repeat_ngram_size=no_repeat_ngram_size, num_beams=num_beams)
            output_text = outputs[0]["generated_text"]
            return output_text

        if args.generation_data_path and args.generation_data_path != "None":
            data = pd.read_csv(args.generation_data_path)
            data = data.rename(columns={args.generation_data_question_column_name: "question"})
            logging.info(f"Data loaded from {args.generation_data_path}")
            logging.info("Processing generation data from CSV")
            for idx, question in tqdm(enumerate(data['question']), total=len(data)):
                try:
                    logging.info(f"Processing index {idx}")
                    prompt = add_default_prompt(question)
                    text_output = generate_output(prompt, max_new_tokens=512, num_beams=3, no_repeat_ngram_size=10)
                    data.at[idx, 'text_output'] = text_output
                    logging.info(f"Question: {question}")
                    logging.info(f"Output: {text_output}")
                except Exception as e:
                    logging.error(f"Error processing index {idx}: {e}")
                    data.at[idx, 'text_output'] = "error"
            # Save results to CSV
            output_path = args.output_path
            data[['question', 'text_output']].to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            logging.info("No generation_data_path provided. Enter questions manually (type 'exit' to quit).")
            while True:
                question = input("질문을 입력하세요 (종료하려면 'exit' 입력): ").strip()
                if question == "" or question.lower() == "exit":
                    print("종료합니다.")
                    break
                prompt = add_default_prompt(question)
                try:
                    text_output = generate_output(prompt, max_new_tokens=512, num_beams=3, no_repeat_ngram_size=10)
                    print(f"답변: {text_output}")
                except Exception as e:
                    print(f"Error: {e}")
                    logging.error(f"Error processing manual input: {e}")
                # data.at[idx, 'text_output'] = text_output
                # logging.info(f"Question: {question}")
                # logging.info(f"Output: {text_output}")
                
                # logging.error(f"Error processing index {idx}: {e}")
                # data.at[idx, 'text_output'] = "error"

        # Save results to CSV
        output_path = args.output_path
        # data[['question', 'text_output']].to_csv(output_path, index=False, encoding="utf-8-sig")

if __name__=="__main__":
    main()
    

