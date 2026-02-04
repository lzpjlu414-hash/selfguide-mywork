import json
import csv
import time
import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser
import re
import ast

from src.llm_client import chat_complete
from src.utils.dataset_io import validate_openai_api_key



def add_message(role, content, history):
    history.append({"role": role, "content": content})


MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-1106")




def ai_request(history, dataset_key, prompt_type, t=0.2, max_retries=3, mock_llm=False, mock_profile=None):
    return chat_complete(
        messages=history,
        model=MODEL,
        temperature=t,
        max_retries=max_retries,
        mock_llm=mock_llm,
        mock_profile=mock_profile,
        dataset_key=dataset_key,
        prompt_type=prompt_type,
    )


def baseline(dataset, start_index=0, mock_llm=False, mock_profile=None):
    validate_openai_api_key(mock_llm=mock_llm)
    print(f"Running baseline for dataset: {dataset}")
    # Create directory for logs if it doesn't exist
    log_dir = f'log/baseline/{dataset}'
    os.makedirs(log_dir, exist_ok=True)

    # Open the JSONL file
    with open(f'data/{dataset}/test.jsonl', 'r', encoding='utf-8') as file:
        # Read all lines from the file
        lines = file.readlines()

        for i, line in tqdm(enumerate(lines[start_index:], start=start_index)):
            if i >= start_index + 1:
                break
            # Parse the JSON data
            data = json.loads(line)

            # Extract data from the JSON object
            sample_id = data.get("id", i)
            question = data.get("question", "")
            answers = data.get("answer", "")

            prompt0 = f"Question: {question} \nAnswer: "

            history = []
            add_message('user', prompt0, history)
            print("DEBUG: before first request")
            output = ai_request(
                history,
                dataset_key=dataset,
                prompt_type="baseline",
                mock_llm=mock_llm,
                mock_profile=mock_profile,
            )
            print("DEBUG: after first request")

            add_message('assistant', output, history)
            time.sleep(1)
            correctness = 'True' if str(answers).strip() in str(output).strip() else 'False'

            # Save log
            log_filename = f'{dataset}_{i}.json'
            log_path = os.path.join(log_dir, log_filename)
            with open(log_path, 'w') as log_file:
                json.dump({
                    "id": sample_id,
                    "question": question,
                    'answer': answers,
                    "correctness": correctness,
                    "log": history,
                }, log_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline script with dataset and start index arguments")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--mock_profile", default=None)
    args = parser.parse_args()

    baseline(
        args.dataset,
        args.start_index,
        mock_llm=args.mock_llm,
        mock_profile=args.mock_profile,
    )


