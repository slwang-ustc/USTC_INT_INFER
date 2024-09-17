import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../llamascratch')))
from llama import LlamaForCausalLM
from utils import init_attention_mask, DynamicCache
import torch
from transformers import AutoTokenizer

class Engine:
    def __init__(self, model_path:str, device) -> None:
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = device
    def execute(self, prompts:list[str], max_new_tokens:int=128, temperature=0.001)->list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt",  padding = True).to(self.device)#type:ignore
        prompt_ids = inputs['input_ids']
        batch_size = prompt_ids.shape[0]
        attention_mask = inputs['attention_mask']
        generated_ids = prompt_ids
        gen_length = 0
        input_len = prompt_ids.shape[1]
        attention_mask = init_attention_mask(batch_size = batch_size, input_len = input_len, padded_mask = attention_mask, device=self.device)
        kvcache = DynamicCache.from_legacy_cache(None)
        with torch.no_grad():
            while True:
                #forward
                logits = self.model(
                    input_ids = prompt_ids,
                    attention_mask = attention_mask,
                    kvcache = kvcache
                    )

                #sample
                logits = logits[:, -1, :].to(torch.float32)
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)             
                samples = torch.multinomial(probs, 1, replacement=True)
                
                new_tokens = samples
                generated_ids = torch.cat((generated_ids, new_tokens), dim=1)

                prompt_ids = new_tokens
                gen_length = gen_length + 1

                if (new_tokens == self.tokenizer.eos_token_id).all() or gen_length >= max_new_tokens:
                    break

                # adjust new attention mask
                cated_mask = torch.zeros((2, 1, 1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((attention_mask[:,:,-1:,:], cated_mask), dim=-1)
        # results might contain padded eos token and excess tokens after eos token
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

