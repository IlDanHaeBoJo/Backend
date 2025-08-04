from openai import OpenAI
import os

## input your open api key here
openai_key = ''
client = OpenAI(api_key=openai_key)

batch_input_file = client.files.create(
  file=open("/workspace/datasets/batch_requests_professional.jsonl", "rb"),
  purpose="batch"
)

batch_input_file_id = batch_input_file.id

res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "nightly eval job"})

print(f"Batch request created with ID: {batch_input_file_id}")
print(res)