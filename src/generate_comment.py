import glob
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Update the checkpoint for GPT-Neo
checkpoint = "EleutherAI/gpt-neo-2.7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add variables for prompts
prompt_start = "This is a piece of code for review: "
prompt_end = "Review complete. Comments: "

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

# Get the path to the src directory
src_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), "src")
src_path = os.path.expandvars(src_path)

# Find all files in the src directory
all_files = glob.glob(os.path.join(src_path, "*"))

# Loop through all files in the src directory
for file in all_files:

    # Check if the file is a regular file
    if os.path.isfile(file):

        # Print the file name
        print(f"Processing file: {file}")

        # Open the file and read the code
        with open(file, "r") as f:
            code = f.read()

        # Construct full prompt
        full_prompt = f"{prompt_start}\n{code}\n{prompt_end}\n"

        # Tokenize the prompt
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

        # Generate a comment for the code
        generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
        tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Print the comment
        print(f"Generated comment: {tokens}")

        # Write the comment to the output file
        with open("src/files/output.txt", "a") as f:
            f.write(f"{tokens}\n\n\n")
