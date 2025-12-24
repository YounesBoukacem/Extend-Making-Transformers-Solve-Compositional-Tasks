import random
import numpy as np
import gc
from tokenizer import TinypyTokenizer

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

SOURCE_DATA_PATH = "./data.txt"
TRAIN_SPLIT = 0.98; TRAIN_DATA_PATH = "./train.txt"; TRAIN_BIN_DATA_PATH = "./train.bin"
VAL_SPLIT = 0.01; VAL_DATA_PATH = "./val.txt"; VAL_BIN_DATA_PATH = "./val.bin"
TEST_DATA_PATH = "./test.txt"
VOCAB_SIZE_PATH = "./vocab_size.txt"

# Initialize the tokenizer
print("[*] Initializing the tokenizer ...")
tpt = TinypyTokenizer()

# Load the dataset of traced snippets
print("[*] Loading the dataset of traced snippets ...")
with open(SOURCE_DATA_PATH, "r") as f:
	data = f.read()

# Split dataset by examples
print("[*] Splitting dataset by examples ...")
examples = data.split("\n\n")[:-1]

# Free memory from the loaded dataset
print("[*] Freeing memory from loaded dataset ...")
del data
gc.collect()

# Print the number of examples
print(f"[*] Total number of examples: {len(examples):,}")

# Set the proportions for train, validation, and test splits
print("[*] Setting the proportions for train, validation, and test splits ...")
train_number = int(len(examples) * TRAIN_SPLIT)
val_number = int(len(examples) * VAL_SPLIT)
test_number = len(examples) - train_number - val_number

# Creating the train dataset
print("[*] Creating the train dataset ...")
train_examples = examples[:train_number]
print(f"[*] There are {len(train_examples)} train examples.")
train_data = "\n\n".join(train_examples) + "\n\n"
del train_examples
print("[*] Writing the train dataset to train.txt")
with open(TRAIN_DATA_PATH, 'w') as f:
	f.write(train_data)
del train_data

# We generate the tokenized file of train.txt in train.bin
print("[*] We generate the tokenized file of train.txt in train.bin")
print(tpt.encode_to_file(TRAIN_DATA_PATH, TRAIN_BIN_DATA_PATH))

# Creating the validation dataset
print("[*] Creating the validation dataset ...")
val_examples = examples[train_number:train_number+val_number]
print(f"[*] There are {len(val_examples)} validation examples.")
val_data = "\n\n".join(val_examples) + "\n\n"
del val_examples
print("[*] writing the validation dataset to val.txt")
with open(VAL_DATA_PATH, 'w') as f:
	f.write(val_data)
del val_data
	
# We generate the tokenized file of val.txt in val.bin
print("[*] We generate the tokenized file of val.txt in val.bin")
print(tpt.encode_to_file(VAL_DATA_PATH, VAL_BIN_DATA_PATH))

# Creating the test dataset
print("[*] Creating the test dataset ...")
test_examples = examples[-test_number:]
print(f"There are {len(test_examples)} test examples")
test_data = "\n\n".join(test_examples) + "\n\n"
del test_examples
print("[*] Writing the test dataset to test.txt")
with open(TEST_DATA_PATH, 'w') as f:
	f.write(test_data)
del test_data

print("[*] Freeing examples memory ...")
del examples
gc.collect()

# Create the vocab_size.txt file
print("[*] Creating the vocab_size.txt file ...")
with open("./vocab_size.txt", "w") as f:
	f.write(str(len(tpt.keywords)))