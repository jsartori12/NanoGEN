#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:48:38 2024

@author: joao
"""

from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List
from tqdm import tqdm

path_model = "/home/joao/.cache/huggingface/hub/models--nferruz--ProtGPT2/snapshots/44255568d9f72bbfa05b23d3826599327ca37910/"

# protgpt2 = pipeline('text-generation', model=path_model)

# sequences = protgpt2("MADVQLQASGGGLVQAGGSLRLSCAASGNINTIDVMGWYRQAPGKQRELVADITRLASAN", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=1, eos_token_id=0, temperature=0.7)



###################3

tokenizer = GPT2Tokenizer.from_pretrained(path_model)
model = GPT2LMHeadModel.from_pretrained(path_model)

input_text = ''.join("M A D V Q L Q A S G G G L V Q A G G S L R L S C A A S G N I N T I D V M G W Y R Q A P G K Q R E L V A D I T R L A S A N Y A D S V K G R F T I S R D N A K N T V Y L Q M N N L E P K D T A V Y Y C A Q W I L S T D H S Y M H Y W G Q G T Q V T V T V S S".split())

cdrs = [25,26,27,28,29,30,31,51,52,53,54,55,94,95,96,97,98,99,100,101,102,103,104,105]

def split_cdrs(cdrs: List[int]) -> List[List[int]]:
    cdrs_split = []
    current_cdr = []
    
    for i in range(len(cdrs)):
        if not current_cdr or cdrs[i] == current_cdr[-1] + 1:
            current_cdr.append(cdrs[i])
        else:
            cdrs_split.append(current_cdr)
            current_cdr = [cdrs[i]]
    cdrs_split.append(current_cdr)  # Add the last group

    return cdrs_split

# Function to create masked_pos list
def create_masked_pos(input_text: str, cdrs: List[int], flank_size: int) -> List[int]:
    cdrs_split = split_cdrs(cdrs)
    
    # Flatten the list of CDRs for easier processing
    all_cdrs = [cdr for sublist in cdrs_split for cdr in sublist]
    
    # Identify the framework regions
    frame_works = []
    for cdr in cdrs_split:
        start_flank = max(0, cdr[0] - flank_size)
        end_flank = min(len(input_text), cdr[-1] + flank_size + 1)
        frame_works.extend(range(start_flank, cdr[0]))
        frame_works.extend(range(cdr[-1] + 1, end_flank))
    
    # Create the masked_pos list
    masked_pos = [1 if i in all_cdrs else 2 if i in frame_works else 0 for i in range(len(input_text))]
    
    return masked_pos, frame_works

def generate_new_tokens(input_seq, n_tokens):
    input_ids = tokenizer.encode(input_seq, return_tensors="pt")

    # Number of tokens to generate
    num_tokens_to_generate = n_tokens

    # Calculate max_length
    max_length = input_ids.shape[1] + num_tokens_to_generate

    # Generate sequences
    sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=950,
        repetition_penalty=1.2,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,  # Use the correct EOS token ID
        temperature=0.7
    )
    
    # Decode the generated sequences
    generated_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
    generated_sequences = generated_sequences[0]
    #input_seq = "".join(list(input_seq))
    # Extract new tokens
    new_tokens = generated_sequences[len(input_seq):]


    return new_tokens

def count_consecutive_twos(start_index, masks):
    count_2 = 0
    for i in range(start_index, len(masks)):
        if masks[i] == 2:
            count_2 += 1
        else:
            break
    return count_2

# def generate_new_sequence(input_text, masks):
#     building_sequence = ""
#     count = 0
    
#     while count != len(input_text):
#         if masks[count] in (0, 1):
#             building_sequence += input_text[count]
#             count += 1
#         elif masks[count] == 2:
#             count_2 = sum(1 for i in masks[count:] if i == 2)
#             print(f"Count de 2 {count_2}")
#             temp_seq = generate_new_tokens(building_sequence, 20)
#             temp_seq = temp_seq.replace("\n", "")
#             building_sequence += temp_seq[:count_2]
#             count += count_2

    
#     return building_sequence


# def generate_new_sequence(input_text, masks):
#     building_sequence = ""
#     count = 0
    
#     while count != len(input_text):
#         if masks[count] in (0, 1):
#             building_sequence += input_text[count]
#             count += 1
#         elif masks[count] == 2:
#             count_2 = count_consecutive_twos(count, masks)
#             print(f"Count de 2 {count_2}")
#             print(f"Building sequence... {building_sequence}")
#             temp_seq = generate_new_tokens(building_sequence, 20)
#             temp_seq = temp_seq.replace("\n", "")
#             print(temp_seq[:count_2])
#             print(f"temp full: {temp_seq}")
#             building_sequence += temp_seq[:count_2]
#             count += count_2

    
#     return building_sequence

def generate_new_sequence(input_text, masks):
    building_sequence = ""
    count = 0
    
    while count != len(input_text):
        if masks[count] in (0, 1):
            building_sequence += input_text[count]
            count += 1
        elif masks[count] == 2:
            count_2 = count_consecutive_twos(count, masks)
            print(f"Building sequence... {building_sequence}")
            temp_seq = generate_new_tokens(building_sequence, 20)
            temp_seq = temp_seq.replace("\n", "")
            building_sequence += temp_seq[:count_2]
            count += len(temp_seq[:count_2])

    
    return building_sequence



# def generate_new_sequence(input_text, masks):
#     building_sequence = ""
#     count = 0
    
#     while count < len(input_text):
#         if masks[count] == 0 or masks[count] == 1:
#             building_sequence += input_text[count]
#             count += 1
#         elif masks[count] == 2:
#             count_2 = sum(1 for i in masks[count:] if i == 2)
#             print(count_2)
            
#             temp_seq = generate_new_tokens(building_sequence, 30)
#             temp_seq = temp_seq.replace("\n", "")
#             print(temp_seq)
#             if len(temp_seq) < count_2:

#                 raise ValueError("Generated sequence is shorter than expected.")
            
#             building_sequence += temp_seq[:count_2]
#             count += count_2
    
#         #print(f"Current count: {count}, Current sequence: {building_sequence}")
    
#     if len(building_sequence) != len(input_text):
#         raise ValueError("Output sequence length does not match input sequence length.")
    
#     return building_sequence

masked_pos, framework = create_masked_pos(input_text = input_text, cdrs = cdrs, flank_size = 4)

initial_input = "".join(input_text[0:framework[0]])



flank_test_list = list(range(1,6))

alignment_file = "aligned_sequences.fasta"

# for flank_size in flank_test_list:
#     masked_pos, framework = create_masked_pos(input_text = input_text, cdrs = cdrs, flank_size = flank_size)
#     for i in range(1,21):
#         new_seq_test = generate_new_sequence(input_text = input_text, masks = masked_pos)
#         with open(alignment_file, "a") as f:
#             f.write(f">Size_{flank_size}_n_{i}\n{new_seq_test}\n")
            



for flank_size in tqdm(flank_test_list, desc="Flank Sizes"):
    masked_pos, framework = create_masked_pos(input_text = input_text, cdrs = cdrs, flank_size = flank_size)
    for i in tqdm(range(1, 21), desc=f"Sequences for flank size {flank_size}", leave=False):
        new_seq_test = generate_new_sequence(input_text = input_text, masks = masked_pos)
        print(f"New sequence len: {len(new_seq_test)}")
        with open(alignment_file, "a") as f:
            f.write(f">Size_{flank_size}_n_{i}\n{new_seq_test}\n")






