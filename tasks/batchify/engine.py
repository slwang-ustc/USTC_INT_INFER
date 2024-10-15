import sys
import os
sys.path.append('..')
from llamascratch.llama import LlamaForCausalLM
from llamascratch.utils import init_attention_mask, Cache, DynamicCache, SinkCache
from llamascratch.utils import top_k_sampling, top_p_sampling, greedy_sampling, Beam, beam_search
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

    def execute(self, prompts:list[str], 
                max_new_tokens:int=128, 
                temperature=0.001,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                beam_size: Optional[int] = None,
                greedy:Optional[bool] = None,
                kvcache:Optional[Cache] = None
                )->list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt",  padding = True).to(self.device)#type:ignore
        prompt_ids = inputs['input_ids']
        batch_size = prompt_ids.shape[0]
        attention_mask = inputs['attention_mask']
        generated_ids = prompt_ids
        gen_length = 0
        input_len = prompt_ids.shape[1]
        attention_mask = init_attention_mask(batch_size = batch_size, input_len = input_len, padded_mask = attention_mask, device=self.device)
        if beam_size is not None:
            assert batch_size == 1, "batch size should be 1"
            batch_size = beam_size
            prompt_ids = prompt_ids.expand(beam_size, -1)
            beams = [Beam(tokens=prompt_ids[0].tolist(), log_prob=0.0)]
            attention_mask = attention_mask.expand(beam_size, -1, -1, -1)
            kvcache = None

        with torch.no_grad():
            while True:
                #forward
                logits = self.model(
                    input_ids = prompt_ids,
                    attention_mask = attention_mask,
                    kvcache = kvcache
                    )
                logits = logits[:, -1, :].to(torch.float32)


                if beam_size is not None:
                    ###维护一个beam的数据结构，每次forward生成logits之后，
                    ###对每个beam进行topk采样，生成新的k个beam
                    beams = beam_search(logits, prompt_ids, beam_size, beams)
                    #更新prompt_ids
                    tokens_list = [beam.tokens for beam in beams]
                    new_tokens = torch.tensor(tokens_list).to(self.device)
                elif top_k is not None:
                    new_tokens = top_k_sampling(logits, k=top_k, temperature=temperature)
                elif top_p is not None:
                    new_tokens = top_p_sampling(logits, p=top_p, temperature=temperature)
                elif greedy is True:
                    new_tokens = greedy_sampling(logits)
                else:
                    # 默认使用温度采样
                    probs = torch.softmax(logits / temperature, dim=-1)
                    new_tokens = torch.multinomial(probs, 1, replacement=True)

                if beam_size is None:
                    generated_ids = torch.cat((generated_ids, new_tokens), dim=1)
                prompt_ids = new_tokens
                gen_length = gen_length + 1
                if (new_tokens == self.tokenizer.eos_token_id).all() or gen_length >= max_new_tokens:
                    break

                # adjust new attention mask
                if beam_size is None:
                    cated_mask = torch.zeros((batch_size, 1, 1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat((attention_mask[:,:,-1:,:], cated_mask), dim=-1)
                else:
                    attention_mask = torch.zeros((beam_size, prompt_ids.shape[-1], prompt_ids.shape[-1]))
                    attention_mask[:, :, :] = torch.triu(torch.full((prompt_ids.shape[-1], prompt_ids.shape[-1]), float('-inf')), diagonal=1)
                    attention_mask = attention_mask[:,None, :, :].to(self.device)

        # results might contain padded eos token and excess tokens after eos token
        if beam_size is not None:
            best_beam = max(beams, key=lambda x: x.log_prob)
            generated_ids = torch.tensor(best_beam.tokens).unsqueeze(0)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)