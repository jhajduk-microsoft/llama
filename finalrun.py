import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os

def load_model(model_name):
    # Use the token from the environment variable if available
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=token)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(model_name, prompt, max_length, output_file):
    model, tokenizer = load_model(model_name)
    output_text = generate_text(model, tokenizer, prompt, max_length)
    print("Generated Text:\n", output_text)
    with open(output_file, 'w') as f:
        f.write(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with Llama2 model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text for the model')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--output_file', type=str, required=True, help='File to save the generated text')
    args = parser.parse_args()
    
    main(args.model_name, args.prompt, args.max_length, args.output_file)

