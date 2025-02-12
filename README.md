# NanoGEN

This project utilizes the Hugging Face Transformers library to design antibody sequences using the PROTGPT2 model. The primary objective is to generate antibody sequences with specific Complementarity-Determining Regions (CDRs) and framework modifications.

## Installation

To run this project, you must install Hugging Face:
<br />
https://huggingface.co/docs/transformers/en/installation

After install hugging face and it dependencies, install the pre-trained model from ProtGPT2:
<br />
https://huggingface.co/nferruz/ProtGPT2/tree/main?library=transformers

## Download the repository in your directory

```
cd your_directory
git clone https://github.com/jsartori12/NanoGEN.git
```

## Running the code
### Loading pre-trained model
First go to the "functions_protgpt2.py" and paste the path of the directory containing the pre-trained model in the variable "path_model" 
<br />

```python


#### Load directory containing binaries for the pre-trained model

path_model = ""

#### Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(path_model)
model = GPT2LMHeadModel.from_pretrained(path_model)
```
### Designing CDRs or frameworks
generate_new_protgpt2.py is used to design CDRs for a given antibody or design frameworks residues flanking the CDRs.
<br/>
Follow below a usage example of the function:
<br/>
```python

####### Loading libraries
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List
from tqdm import tqdm
import functions_protgpt2


#### Input antibody sequence
input_text = "MADVQLQASGGGLVQAGGSLRLSCAASGNINTIDVMGWYRQAPGKQRELVADITRLASANYADSVKGRFTISRDNAKNTVYLQMNNLEPKDTAVYYCAQWILSTDHSYMHYWGQGTQVTVTVSS"


#### List of CDRs indexs
cdrs = [25,26,27,28,29,30,31,51,52,53,54,55,94,95,96,97,98,99,100,101,102,103,104,105]

#### Designing CDRs residues for a given antibody sequence
design_cdrs = functions_protgpt2.Create_sequences(input_text = input_text, 
                                                cdrs_list = cdrs,
                                                design_type = "cdr")

#### Designing 3 framework residues flanking the CDRs for a given antibody sequence
design_framework = functions_protgpt2.Create_sequences(input_text = input_text, 
                                                cdrs_list = cdrs,
                                                design_type = "fm",
                                                flank_size = 3)


```







