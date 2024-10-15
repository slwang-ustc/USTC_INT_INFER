import sys
sys.path.append('..')
from batchify.engine import Engine
from llamascratch.utils import DynamicCache, SinkCache
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
DEVICE = 'cuda'
weight_path = os.getenv("HF_MODEL_PATH")
if weight_path is None:
    raise ValueError("HF_MODELS_CACHE environment variable is not set")

engine = Engine(weight_path, DEVICE)
batch = ["Hello how is it going?", "Is this a cat?"]

#dynamic cache + top k
dynamiccache = DynamicCache.from_legacy_cache(None)
result = engine.execute(prompts=batch, top_k = 3, kvcache=dynamiccache)
print(f"top_k -3 result: {result}")

#sink cache + top p
sinkcache = SinkCache.from_legacy_cache(None)
result = engine.execute(prompts=batch, top_p = 0.8, kvcache=sinkcache)
print(f"top_p -0.8 result: {result}")

#dynamic cache + temperature
dynamiccache = DynamicCache.from_legacy_cache(None)
result = engine.execute(prompts=batch, kvcache=dynamiccache)
print(f"temperature result: {result}")

#sink cache + greedy
sinkcache = SinkCache.from_legacy_cache(None)
result = engine.execute(prompts=batch, greedy=True, kvcache=sinkcache)
print(f"greedy result: {result}")

#beam search without cache
result = engine.execute(prompts=["Hello how is it going?"], beam_size=3)
print(f"beam result: {result}")