# NanoGEN

This project utilizes the Hugging Face Transformers library to design antibody sequences using ESM2 model. The primary objective is to generate antibody sequences with specific Complementarity-Determining Regions (CDRs) and framework modifications.

## Installation

Setting up the environment using the .yml:
<br />

1. Clone the repository:
    ```sh
    git clone https://github.com/jsartori12/NanoGEN.git
    cd NanoGEN
    ```
2. Create the environment:
    ```sh
    conda env create -f environment.yaml
    ```
3. Activate the environment:
    ```sh
    conda activate esmfold
    ```

## Running the code

### Designing Options
main.py contain 3 usage examples for generate from the entire sequence, only for CDRs and only for frameworks

<br />

```python
from functions_esm2 import generate_Sequence

input_sequence = "MADVQLQASGGGLVQAGGSLRLSCAASGNINTIDVMGWYRQAPGKQRELVADITRLASANYADSVKGRFTISRDNAKNTVYLQMNNLEPKDTAVYYCAQWILSTDHSYMHYWGQGTQVTVTVSS"

cdrs = [25, 26, 27, 28, 29, 30, 31, 51, 52, 53, 54, 55, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

#### Generate for all positions
all_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="all")
#### Generate for CDRs
CDRs_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="cdr")
#### Generate for frameworks
FM_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="fm")

```

# Animation showing the evolution of a Nanobody sequence




