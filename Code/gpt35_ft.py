import json
import tiktoken
import numpy as np
from collections import defaultdict
import os
from openai import AzureOpenAI
from IPython.display import clear_output
import time

def prelim():
    # Load the training set
    with open('train_implicit_all.jsonl', 'r', encoding='utf-8') as f:
        training_dataset = [json.loads(line) for line in f]

    # Training dataset stats
    print("Number of examples in training set:", len(training_dataset))
    print("First example in training set:")
    for message in training_dataset[0]["messages"]:
        print(message)

    # Load the validation set
    with open('dev_implicit_all.jsonl', 'r', encoding='utf-8') as f:
        validation_dataset = [json.loads(line) for line in f]

    # Validation dataset stats
    print("\nNumber of examples in validation set:", len(validation_dataset))
    print("First example in validation set:")
    for message in validation_dataset[0]["messages"]:
        print(message)
        
        

    encoding = tiktoken.get_encoding("cl100k_base") # default encoding used by gpt-4, turbo, and text-embedding-ada-002 models

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    files = ['data_ft/training_set.jsonl', 'data_ft/validation_set.jsonl']

    for file in files:
        print(f"Processing file: {file}")
        with open(file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        total_tokens = []
        assistant_tokens = []

        for ex in dataset:
            messages = ex.get("messages", {})
            total_tokens.append(num_tokens_from_messages(messages))
            assistant_tokens.append(num_assistant_tokens_from_messages(messages))

        print_distribution(total_tokens, "total tokens")
        print_distribution(assistant_tokens, "assistant tokens")
        print('*' * 50)
        
        
        
def train():

    client = AzureOpenAI(azure_endpoint = "", 
    api_key="",  
    api_version="")

    training_file_name = 'data_ft/training_set.jsonl'
    validation_file_name = 'data_ft/validation_set.jsonl'

    # Upload the training and validation dataset files to Azure OpenAI with the SDK.

    training_response = client.files.create(
        file = open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id

    validation_response = client.files.create(
        file = open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)
    
    
    # Submit fine-tuning training job

    response = client.fine_tuning.jobs.create(
        training_file = training_file_id,
        validation_file = validation_file_id,
        model = "gpt-35-turbo-0613", # Enter base model name. Note that in Azure OpenAI the model name contains dashes and cannot contain dot/period characters.
    )

    job_id = response.id



    print("Job ID:", response.id)
    print("Status:", response.status)
    print(response.model_dump_json(indent=2))
    # Track training status

    start_time = time.time()

    # Get the status of our fine-tuning job.
    response = client.fine_tuning.jobs.retrieve(job_id)

    status = response.status

    # If the job isn't done yet, poll it every 10 seconds.
    while status not in ["succeeded", "failed"]:
        time.sleep(10)

        response = client.fine_tuning.jobs.retrieve(job_id)
        print(response.model_dump_json(indent=2))
        print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
        status = response.status
        print(f'Status: {status}')
        clear_output(wait=True)

    print(f'Fine-tuning job {job_id} finished with status: {status}')

    # List all fine-tuning jobs for this resource.
    print('Checking other fine-tune jobs for this resource.')
    response = client.fine_tuning.jobs.list()
    print(f'Found {len(response.data)} fine-tune jobs.')
