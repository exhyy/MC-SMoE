from transformers import AutoConfig, AutoTokenizer, MixtralForCausalLM, MixtralConfig
import torch

def main():
    path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1'
    model = MixtralForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
    )
    assert isinstance(model, MixtralForCausalLM)

    config = AutoConfig.from_pretrained(path)
    assert isinstance(config, MixtralConfig)
    config.num_experts_per_tok = 4
    model.config = config

    print('saving model')
    model.save_pretrained('./Mixtral-8x7B-Instruct-v0.1_top4')

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.save_pretrained('./Mixtral-8x7B-Instruct-v0.1_top4')
    print('ok')


if __name__ == '__main__':
    main()
