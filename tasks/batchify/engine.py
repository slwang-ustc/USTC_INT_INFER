import sys
import os
sys.path.append('..')
from llamascratch.llama import LlamaForCausalLM
from llamascratch.utils import init_attention_mask, DynamicCache
import torch
from transformers import AutoTokenizer
from typing import Optional

class Engine:
    def __init__(self, model_path:str, device) -> None:
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = device
    def top_k_sampling(self, logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:

        logits = logits / temperature
        
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

        probs = torch.softmax(mask, dim=-1)

        samples = torch.multinomial(probs, num_samples=1)

        return samples
    def top_p_sampling(self, logits: torch.Tensor, p: float, temperature: float) -> torch.Tensor:

        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_keep = cumulative_probs <= p

        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = 1

        sorted_probs = sorted_probs * sorted_indices_to_keep.float()

        # 重新归一化
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        samples = torch.multinomial(probs, num_samples=1)

        return samples
    def execute(self, prompts:list[str], 
                max_new_tokens:int=128, 
                temperature=0.001,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                beam_size: Optional[int] = None
                )->list[str]:
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
                logits = logits[:, -1, :].to(torch.float32)

                # 根据采样策略进行采样
                if top_k is not None:
                    samples = self.top_k_sampling(logits, k=top_k, temperature=temperature)
                elif top_p is not None:
                    samples = self.top_p_sampling(logits, p=top_p, temperature=temperature)
                else:
                    # 默认使用温度采样
                    probs = torch.softmax(logits / temperature, dim=-1)
                    samples = torch.multinomial(probs, 1, replacement=True)
                
                new_tokens = samples
                generated_ids = torch.cat((generated_ids, new_tokens), dim=1)

                prompt_ids = new_tokens
                gen_length = gen_length + 1

                if (new_tokens == self.tokenizer.eos_token_id).all() or gen_length >= max_new_tokens:
                    break

                # adjust new attention mask
                cated_mask = torch.zeros((batch_size, 1, 1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((attention_mask[:,:,-1:,:], cated_mask), dim=-1)
        # results might contain padded eos token and excess tokens after eos token
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

