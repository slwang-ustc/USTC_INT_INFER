import torch
import torch.nn as nn

import math



def repeat_kv(states, n_repeat):
    batch_size, seq_len, num_kv_heads, head_dim = states.shape
    if n_repeat == 1:
        return states
    return states[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_repeat, head_dim).reshape(batch_size, seq_len, num_kv_heads * n_repeat, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, bias) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
    
    def forward(self, x):
        batch_size, sequence_len, hidden_size = x.shape
        
        q_states = self.q_proj(x)  # bsz, seqlen, num_heads * head_dim
        k_states = self.k_proj(x)  # bsz, seqlen, num_kv_heads * head_dim
        v_states = self.v_proj(x)  # bsz, seqlen, num_kv_heads * head_dim
        
        q_states = q_states.reshape(batch_size, sequence_len, self.num_heads, self.head_dim)
        k_states = k_states.reshape(batch_size, sequence_len, self.num_kv_heads, self.head_dim)
        v_states = v_states.reshape(batch_size, sequence_len, self.num_kv_heads, self.head_dim)
        
        k_states = repeat_kv(k_states, self.num_groups)  # bsz, seqlen, num_heads, head_dim
        v_states = repeat_kv(v_states, self.num_groups)  # bsz, seqlen, num_heads, head_dim
        
        q_states = q_states.transpose(1, 2)  # bsz, num_heads, seqlen, head_dim
        k_states = k_states.transpose(1, 2).transpose(2, 3)  # bsz, num_heads, head_dim, seqlen
        v_states = v_states.transpose(1, 2)  # bsz, num_heads, seqlen, head_dim
        
        attention_scores = torch.matmul(q_states, k_states) / math.sqrt(self.head_dim)  # bsz, num_heads, seqlen, seqlen
        attention_scores = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(q_states.dtype)
        
        attention_output = torch.matmul(attention_scores, v_states)  # bsz, num_heads, seqlen, head_dim
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_len, -1)  # bsz, seqlen, num_heads * head_dim
        attention_output = self.o_proj(attention_output)
        
        return attention_output
        


class Engine:
    def __init__(self, model_path:str) -> None:
        ...
    def execute(self, prompts:list[str], max_new_tokens:int=128, temperature=0.001)->list[str]:
        ...

