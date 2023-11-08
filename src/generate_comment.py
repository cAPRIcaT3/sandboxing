import glob
import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Update the checkpoint for GPT-Neo
checkpoint = "EleutherAI/gpt-neo-2.7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add variables for prompts
prompt_start = "This is a piece of code for review: "
prompt_end = "Review complete. Comments: "

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPTNeoForCausalLM.from_pretrained(checkpoint).to(device)

# Get the path to the src directory
src_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), "src")
src_path = os.path.expandvars(src_path)

# Define the output directory
output_dir = os.path.join(src_path, "files")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

output_file = os.path.join(output_dir, "output.txt")

# Open the output file outside the loop
with open(output_file, "a") as output:
    # Loop through all files in the src directory
    for file in glob.glob(os.path.join(src_path, "*")):
        if os.path.isfile(file):
            print(f"Processing file: {file}")
            with open(file, "r") as f:
                code = f.read()

            full_prompt = f"{prompt_start}\n{code}\n{prompt_end}\n"
            input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

            generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
            tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print(f"Generated comment: {tokens}")

            # Write the comment to the output file
            output.write(f"{tokens}\n\n\n")
