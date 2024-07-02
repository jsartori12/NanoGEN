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
import functions_protgpt2


#### Input antibody sequence
input_text = "MADVQLQASGGGLVQAGGSLRLSCAASGNINTIDVMGWYRQAPGKQRELVADITRLASANYADSVKGRFTISRDNAKNTVYLQMNNLEPKDTAVYYCAQWILSTDHSYMHYWGQGTQVTVTVSS"

#### List of CDRs indexs
cdrs = [25,26,27,28,29,30,31,51,52,53,54,55,94,95,96,97,98,99,100,101,102,103,104,105]



design_framework = functions_protgpt2.Create_sequences(input_text = input_text, 
                                                cdrs_list = cdrs,
                                                design_type = "fm",
                                                flank_size = 5)



design_cdrs = functions_protgpt2.Create_sequences(input_text = input_text, 
                                                cdrs_list = cdrs,
                                                design_type = "cdr")







