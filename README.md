# Fine-Tuning FLAN-T5 with LoRA

This repository contains a Jupyter Notebook for fine-tuning the **FLAN-T5 Large** model using **LoRA (Low-Rank Adaptation)**.  
The workflow demonstrates efficient fine-tuning with attention masks and padding to handle variable-length text data.


## Main Libraries used:
  - transformers
  - torch
  - scikit-learn
  - pandas
  - numpy

## Description

The notebook covers:
- **Importing libraries** and setting up GPU/MPS acceleration if available.  
- **Data preparation**: loading the dataset (`Q2_20230202_majority 1.csv`), preprocessing, label encoding, and splitting into train/test sets.  
- **Tokenization & Padding**:  
  - Tokenized input text with Hugging Face’s T5 tokenizer.  
  - Applied **attention masks** to prevent tokens other than those associated with the labels to be used in the loss function 
  - Used padding tokens for sequence alignment.  
- **LoRA Fine-Tuning**:  
  - Implemented custom `LoRALayer` and `LinearWithLoRA` modules.  
  - Replaced linear layers in FLAN-T5 with LoRA-enabled layers (`replace_linear_with_lora`).  
  - Froze all original model parameters — only LoRA parameters were trained.  
- **Training Loop**:  
  - Batched data using PyTorch `DataLoader`.  
  - Forward pass with attention masks applied.  
  - Optimized only LoRA parameters for efficient fine-tuning.  
- **Evaluation**:  
  - Model evaluated on the test split.  
  - Accuracy tracked across epochs.  

## Results

The notebook demonstrates how **LoRA drastically reduces the number of trainable parameters** compared to full fine-tuning, while still adapting FLAN-T5 effectively.  
It also shows how **padding and attention masks** help stabilize training with variable-length inputs and for classification respectively.





