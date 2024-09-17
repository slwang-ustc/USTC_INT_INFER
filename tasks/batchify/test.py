from engine import Engine
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
DEVICE = 'cuda'
weight_path = os.getenv("HF_MODEL_PATH")
if weight_path is None:
    raise ValueError("HF_MODELS_CACHE environment variable is not set")

engine = Engine(weight_path, DEVICE)
batch = ["Hello how is it going?", "Is this a cat?"]
result = engine.execute(batch)

print(result)

