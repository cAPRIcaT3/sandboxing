import glob
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

# Define the pad token ID
pad_token_id = 0

# Define the prompt
prompt = "# this is code for code review, please review it and provide feedback."

# Get the path to the src directory
src_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), "src")
src_path = os.path.expandvars(src_path)

# Define the output directory
output_dir = os.path.join(src_path, "files")
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

output_file = os.path.join(output_dir, "output.txt")

# Open the output file outside the loop
with open(output_file, "a") as output:
    # Loop through all files in the src directory
    for file in glob.glob(os.path.join(src_path, "*")):
        if os.path.isfile(file):
            print(f"Processing file: {file}")
            with open(file, "r") as f:
                code = f.read()

            # Generate comment using the model
            inputs = tokenizer(prompt + code, return_tensors="pt")
            # Print the shape of the inputs tensor
            print(inputs.input_ids.shape)


            # Create an attention mask
            attention_mask = torch.ones(inputs.input_ids.shape, dtype=torch.long, device=inputs.input_ids.device)

            # Generate comments with attention mask
            generated_comments = model.generate(inputs, attention_mask=attention_mask)

            decoded_comments = tokenizer.decode(generated_comments[0], skip_special_tokens=True)
            print(f"Generated comment: {decoded_comments}")

            # Write the comment to the output file
            output.write(f"{decoded_comments}\n\n\n")
