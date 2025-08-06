import subprocess
import logging
from datetime import datetime

def main():
    # Configuration
    evaluation_script = "evaluate_generation.py"
    model_path = "./results/qwen_14b_professional_lora_cot_v2/checkpoint-28000"
    output_path = "./results/test_result_KorMedMCQA.csv"
    mode = "test"    
    
    # Log setup
    log_file = "test_execution_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Log start
    logging.info("Starting test execution.")
    command = (
        f"python {evaluation_script} "
        f"--model_name_or_path='{model_path}' "
        f"--output_path='{output_path}' "
        f"--mode='{mode}' "       
        f"--log_file='{log_file}'"
    )
    logging.info(f"Execution command: {command}")

    # Record start time
    start_time = datetime.now()
    logging.info(f"Process started at: {start_time}")

    # Run the command
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        logging.info("Process completed successfully.")
        logging.info(f"Process output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error("Process failed.")
        logging.error(f"Error output:\n{e.stderr}")
        raise
    finally:
        # Record end time
        end_time = datetime.now()
        logging.info(f"Process ended at: {end_time}")
        logging.info(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    main()
