import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def main():
    path = '/opt/data/private/hyy/MC-SMoE/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group2'
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        device_map='auto',
    )
    print(f'{model.num_parameters()=}')
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompt = 'The meaning of life is'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.cuda()

    outputs = model.generate(input_ids, max_new_tokens=128, do_sample=False)[0]
    text = tokenizer.decode(outputs)
    print(f'output: {text}')

if __name__ == '__main__':
    main()
