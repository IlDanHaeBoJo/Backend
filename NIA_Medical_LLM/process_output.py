import pandas as pd
import re
import argparse
import logging

def process_raw_output(raw_text, truncation_length):
    # Extract text after '답변:'
    match = re.search(r'답변:\s*(.*)', raw_text, re.DOTALL)
    extracted_text = "답변: " + match.group(1).strip() if match else raw_text

    # Remove any second question and additional context starting with specific phrases
    extracted_text = re.sub(r'(당신이 전문|당신은 전문|당신이 의료|다음 질문|질문:).*', '', extracted_text, flags=re.DOTALL).strip()

    # Truncate to the given length
    return extracted_text[:truncation_length]

def main(raw_csv_path, input_csv_path, truncation_length):
    # Load the two CSV files
    raw_df = pd.read_csv(raw_csv_path)
    input_df = pd.read_csv(input_csv_path)

    # Check that both CSVs have the same number of rows
    if len(raw_df) != len(input_df):
        raise ValueError("The two input CSV files must have the same number of rows.")

    # Process the raw generation output
    processed_outputs = raw_df['text_output'].apply(lambda x: process_raw_output(x, truncation_length))

    # Add the processed output as a new column in input CSV
    input_df['Qwen 답변'] = processed_outputs

    # Save the updated input CSV (overwriting the input file)
    input_df.to_csv(input_csv_path, index=False, encoding="utf-8-sig")
    print(f"Processed CSV saved to {input_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw generation results and add them to input CSV.")
    parser.add_argument("--raw_csv_path", type=str, help="Path to the raw generation result CSV file.")
    parser.add_argument("--input_csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--truncation_length", type=int, default=512, help="Maximum length to truncate the processed output.")
    parser.add_argument("--log_file", default="test_execution_log.txt", type=str)

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("Processing outputs...")

    # Run the script
    main(args.raw_csv_path, args.input_csv_path, args.truncation_length)

    logging.info("Processing done")
    logging.info("Results saved to input CSV")
