class Engine:
    def __init__(self, model_path:str) -> None:
        ...
    def execute(self, prompts:list[str], max_new_tokens:int=128, temperature=0.001)->list[str]:
        ...

