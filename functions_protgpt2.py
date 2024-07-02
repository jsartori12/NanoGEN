#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:30:02 2024

@author: joao.sartori
"""

from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List
from tqdm import tqdm


#### Load directory containing binaries for the pre-trained model

path_model = "/home/joao.sartori/.cache/huggingface/hub/models--nferruz--ProtGPT2/snapshots/44255568d9f72bbfa05b23d3826599327ca37910/"

#### Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(path_model)
model = GPT2LMHeadModel.from_pretrained(path_model)

def split_cdrs(cdrs: List[int]) -> List[List[int]]:
    """
    Splits a list of CDR positions into consecutive groups.

    Parameters:
    cdrs (List[int]): A list of integers representing CDR positions.

    Returns:
    List[List[int]]: A list of lists, where each sublist contains consecutive CDR positions.
    """
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

def create_masked_pos(input_text: str, cdrs: List[int], flank_size: int) -> List[int]:
    """
    Creates a list indicating the positions to mask based on CDRs and flanking regions.

    Parameters:
    input_text (str): The input text sequence.
    cdrs (List[int]): A list of integers representing CDR positions.
    flank_size (int): The size of the flanking region around each CDR.

    Returns:
    Tuple[List[int], List[int]]: A tuple containing the masked positions list and the framework regions list.
    """
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

def generate_new_tokens(input_seq: str, n_tokens: int) -> str:
    """
    Generates new tokens based on the input sequence using a pre-trained model.

    Parameters:
    input_seq (str): The input sequence to continue generating from.
    n_tokens (int): The number of new tokens to generate.

    Returns:
    str: The generated new tokens.
    """
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
    
    # Extract new tokens
    new_tokens = generated_sequences[len(input_seq):]

    return new_tokens

def count_consecutive_twos(start_index: int, masks: List[int]) -> int:
    """
    Counts the number of consecutive 2s in the masks list starting from a given index.

    Parameters:
    start_index (int): The starting index in the masks list.
    masks (List[int]): The list of mask values.

    Returns:
    int: The count of consecutive 2s.
    """
    count_2 = 0
    for i in range(start_index, len(masks)):
        if masks[i] == 2:
            count_2 += 1
        else:
            break
    return count_2

def generate_new_sequence(input_text: str, masks: List[int]) -> str:
    """
    Generates a new sequence based on the input text and mask positions.

    Parameters:
    input_text (str): The input text sequence.
    masks (List[int]): A list indicating the positions to mask.

    Returns:
    str: The generated new sequence.
    """
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

def Create_sequences(input_text: str, cdrs_list: List[int], design_type: str, flank_size = 1) -> str:
    """
    Creates sequences based on the specified design type.

    Parameters:
    input_text (str): The input text sequence.
    cdrs_list (List[int]): A list of integers representing CDR positions.
    design_type (str): The design type, either "fm" for framework masking or "cdr" for CDR masking.
    flank_size (int): The size of the flanking region around each CDR.

    Returns:
    str: The designed sequence.
    """
    if design_type == "fm":
        masked_pos, framework = create_masked_pos(input_text=input_text, cdrs=cdrs_list, flank_size=flank_size)
        designed_fm = generate_new_sequence(input_text=input_text, masks=masked_pos)
        return designed_fm
    if design_type == "cdr":
        masked_cdrs = [2 if i in cdrs_list else 0 for i in range(len(input_text))]
        designed_cdrs = generate_new_sequence(input_text=input_text, masks=masked_cdrs)
        return designed_cdrs

        
        
        