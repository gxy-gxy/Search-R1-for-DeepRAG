# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
import json
import argparse

import pandas as pd


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> <f/answer> tags, for example <answer> Beijing </answer>. \
Question: {question}\n"""

    elif template_type == 'deeprag':
        prefix = f"""Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model. Your task is to answer questions by logically decomposing them into clear sub-questions and iteratively addressing each one. Use "Follow up:" to introduce each sub-question and "Intermediate answer:" to provide answers. For each sub-question, decide whether you can provide a direct answer or if additional information is required. If additional information is needed, state, "Let's search the question in Wikipedia." and then use the retrieved information to respond comprehensively. If a direct answer is possible, provide it immediately without searching.
Question: {question}
Follow up: """

    else:
        raise NotImplementedError    
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq')
    parser.add_argument('--template_type', type=str, default='deeprag')
    args = parser.parse_args()
    data_source = 'nq'

    files = ['/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/deeprag_checkpoint/data/wikihop/dpo_new.jsonl','/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/deeprag_checkpoint/data/hotpot/dpo_new.jsonl']
    # Load and process JSON data
    train_data = []
    for file in files:
        with open(file, 'r') as f:
            train_data.extend([json.loads(line) for line in f])

    # Process each item in the data
    processed_data = []
    for idx, example in enumerate(train_data):
        example['question'] = example['question'].strip()
        if example['question'][-1] != '?':
            example['question'] += '?'
        
        question = make_prefix(example, template_type=args.template_type)
        solution = {
            "target": example['answer'],
        }

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
            }
        }
        processed_data.append(data)

    # Save to JSON instead of parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    
    print(processed_data[0])
    
    # Convert processed data to DataFrame and save as parquet
    df = pd.DataFrame(processed_data)
    output_path = os.path.join(local_dir, 'train.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f"Saved {len(processed_data)} records to {output_path}")
