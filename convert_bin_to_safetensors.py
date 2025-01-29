import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


def main():
    model_path = '/opt/data/private/hyy/MC-SMoE/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group2_bin'
    output_dir = '/opt/data/private/hyy/MC-SMoE/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group2'

    print(f'load model from {model_path}')
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    print(f'saving model to {output_dir}')
    model.save_pretrained(output_dir)

    print(f'saving tokenizer to {output_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)

    print('ok')

if __name__ == '__main__':
    main()
