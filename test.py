from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("model/PersonalGpt")
model = GPT2LMHeadModel.from_pretrained("model/PersonalGpt")

# Function to generate chatbot response
def generate_response(user_input ):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids,  num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = generate_response(user_input)
    print("Chatbot:", response)
