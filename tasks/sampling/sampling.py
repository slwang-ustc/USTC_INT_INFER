import torch
from cache import DynamicCache

def temperature_sampling_generate(model, input_ids, max_new_tokens=10, temperature=1.0, attention_mask=None):
    device = input_ids.device
    generated = input_ids
    #position_ids = torch.arange(generated.size(1), device=device).unsqueeze(0)
    cache_position = torch.arange(input_ids.size(1), device=device)
    #past_key_values = None  # 初始化past_key_values为空
    past_key_values = DynamicCache.from_legacy_cache(None)
    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=device)

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=generated,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            #position_ids=position_ids,
            cache_position=cache_position,
            device=device,
            use_cache=True  #显式指定使用缓存
        )
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[0][:, -1, :]
        logits = logits / temperature  #使用温度参数调整logits
        probs = torch.nn.functional.softmax(logits, dim=-1)  #计算softmax概率
        next_token = torch.multinomial(probs, num_samples=1)  #根据概率随机选择下一个词
        generated = torch.cat((generated, next_token), dim=1)  #更新生成的序列
        #position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)  #更新position_ids

        num_new_tokens = 1
        cache_position = cache_position[-1:] + num_new_tokens

        # 更新past_key_values为下一次生成使用
        # 这里其实有点点不准确，只有在return_dict为True时才行
        if outputs.past_key_values is not None:
            past_key_values = outputs.past_key_values
        

        if torch.all(next_token == model.config.eos_token_id):
            break

    return generated

def greedy_sampling_generate(model, input_ids, max_new_tokens=10, attention_mask=None):
    device = input_ids.device
    generated = input_ids
    cache_position = torch.arange(input_ids.size(1), device=device)
    past_key_values = DynamicCache.from_legacy_cache(None)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=device)

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=generated,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            device=device,
            use_cache=True  #显式指定使用缓存
        )
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[0][:, -1, :]
        
        
        next_token = torch.argmax(logits, dim=-1, keepdim=True)   #使用贪婪采样选择最高概率的词
        
        generated = torch.cat((generated, next_token), dim=1)  #更新生成的序列

        num_new_tokens = 1
        cache_position = cache_position[-1:] + num_new_tokens

        if outputs.past_key_values is not None:   #更新past_key_values为下一次生成使用
            past_key_values = outputs.past_key_values

        if torch.all(next_token == model.config.eos_token_id):
            break

    return generated


def top_k_sampling_generate(model, input_ids, max_new_tokens=10, top_k=50, attention_mask=None):
    device = input_ids.device
    generated = input_ids
    cache_position = torch.arange(input_ids.size(1), device=device)
    past_key_values = DynamicCache.from_legacy_cache(None)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=device)

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=generated,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            device=device,
            use_cache=True  #显式指定使用缓存
        )
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[0][:, -1, :]
        
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)  #只保留top-k个最高概率的 token
        
        top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)  #计算top-k的softmax 概率

        
        next_token_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)  #从top-k中随机采样下一个token
        next_token = top_k_indices.gather(1, next_token_indices.unsqueeze(-1))
        #print(next_token)
        
        generated = torch.cat((generated, next_token), dim=1)  # 更新生成的序列

        num_new_tokens = 1
        cache_position = cache_position[-1:] + num_new_tokens

        if outputs.past_key_values is not None:  #更新past_key_values为下一次生成使用
            past_key_values = outputs.past_key_values

        if torch.all(next_token == model.config.eos_token_id):
            break

    return generated



def top_p_sampling_generate(model, input_ids, max_new_tokens=10, top_p=0.9, attention_mask=None):
    device = input_ids.device
    generated = input_ids
    cache_position = torch.arange(input_ids.size(1), device=device)
    past_key_values = DynamicCache.from_legacy_cache(None)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=device)

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=generated,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            device=device,
            use_cache=True  # 显式指定使用缓存
        )
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[0][:, -1, :]
        
        probs = torch.nn.functional.softmax(logits, dim=-1)  #对logits进行softmax计算得到概率
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)  #对概率进行排序

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  #计算累计概率

        sorted_indices_to_remove = cumulative_probs > top_p  #找到cumulative_probs中大于top_p的最小索引，保留该索引之前的所有token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  #确保至少有一个token
        sorted_indices_to_remove[..., 0] = False  #保留第一个token

        sorted_probs[sorted_indices_to_remove] = 0
        new_probs = torch.zeros_like(probs)
        new_probs.scatter_(1, sorted_indices, sorted_probs)  # 使用scatter_将过滤后的概率放回到原始的probs数组

        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)  #重新归一化概率

        next_token = torch.multinomial(new_probs, num_samples=1)  #从经过top-p过滤的概率中随机采样下一个token
        generated = torch.cat((generated, next_token), dim=1)  #更新生成的序列

        num_new_tokens = 1
        cache_position = cache_position[-1:] + num_new_tokens

        if outputs.past_key_values is not None:
            past_key_values = outputs.past_key_values
        
        if torch.all(next_token == model.config.eos_token_id):
            break

    return generated


import heapq

class BeamSearchNode(object):
    def __init__(self, sequence, log_prob, past_key_values):
        self.sequence = sequence  #当前的序列
        self.log_prob = log_prob  # 累积的log概率
        self.past_key_values = past_key_values  #该Beam的KV-Cache

    def __lt__(self, other):   #比较两个节点，堆中会根据log_prob排序
        return self.log_prob > other.log_prob  # 在堆中保存log_prob大的节点

def beam_search_generate(model, input_ids, max_new_tokens=10, attention_mask=None, beam_size=2, eos_token_id=None, early_stopping=False):
    device = input_ids.device
    cache_position = torch.arange(input_ids.size(1), device=device)
    
    initial_past_key_values = DynamicCache.from_legacy_cache(None)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=device)

    beam_heap = []
    initial_node = BeamSearchNode(input_ids, 0.0, initial_past_key_values)
    heapq.heappush(beam_heap, initial_node)
    
    completed_sequences = []  # 用于存储生成结束的序列

    for _ in range(max_new_tokens):
        new_beam_heap = []
        #遍历当前堆中的每个序列并扩展
        while beam_heap:
            node = heapq.heappop(beam_heap)  #取出堆中当前概率最高的序列
            seq, log_prob, node_past_key_values = node.sequence, node.log_prob, node.past_key_values

            model_inputs = model.prepare_inputs_for_generation(
                input_ids=seq,
                past_key_values=node_past_key_values,  
                attention_mask=attention_mask,
                cache_position=cache_position,
                device=device,
                use_cache=True 
            )
            outputs = model(**model_inputs)
            logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[0][:, -1, :]
            
            next_token_scores = torch.nn.functional.log_softmax(logits, dim=-1)  #对logits进行softmax以获得概率分布，然后取top-k
            topk_scores, topk_tokens = torch.topk(next_token_scores, beam_size, dim=-1)

            for i in range(beam_size):  #扩展当前节点，生成新的候选序列并推入堆中
                next_token = topk_tokens[0, i].unsqueeze(0) 
                next_log_prob = log_prob + topk_scores[0, i].item()  
                new_sequence = torch.cat([seq, next_token.unsqueeze(0)], dim=1) 

                new_past_key_values = outputs.past_key_values.clone()
                new_node = BeamSearchNode(new_sequence, next_log_prob, new_past_key_values)
                heapq.heappush(new_beam_heap, new_node)

                if next_token == eos_token_id:
                    completed_sequences.append((new_sequence, next_log_prob))

                    if early_stopping:  #如果early_stopping启用，并且完成了序列，则停止该分支的扩展
                        continue

        beam_heap = heapq.nlargest(beam_size, new_beam_heap)  #将新的堆按log_prob排序，并只保留beam_size个节点
        
        del new_beam_heap  # 释放旧的缓存，确保不再使用的节点被回收
        
        num_new_tokens = 1
        cache_position = cache_position[-1:] + num_new_tokens

        if not beam_heap:
            break

    if completed_sequences:  #如果有完成的序列，则返回最高分的完成序列
        best_sequence = max(completed_sequences, key=lambda x: x[1])
        return best_sequence[0]
    
    return beam_heap[0].sequence  #如果没有完成的序列，则返回堆中最高分的序列
