## 这是什么？

这是[USTC-INT](https://int-ustc.github.io/) 的大模型推理入门练习仓库。我们会在这里（main分支）发布任务，您需要向自己的分支提交代码。

任务要求是**必须**完成的内容; 额外要求是可选项. 

## Tasks

### 从零Llama

该任务要求您从头完成Llama模型的构建。

仅允许的依赖：torch, transformers（仅允许使用其from_pretrained加载函数）

检查`tasks/llamascratch`获取更多信息。

### 推理Batch

该任务要求您执行对一个Batch的推理.

在之前的代码的基础上, 修改其, 使其可以推理不同长度的句子.

检查`tasks/batchify`获取更多信息。

### Sampling

该任务要求您完成各种采样核.

检查`tasks/sampling`获取更多信息。

### Rope

该任务将Rope操作从Attention解耦, 允许更灵活的Cache策略.

检查`tasks/rope`获取更多信息。

## Adavanced Topics

本节是各种高级技术的一个小指引. 完成上面的任务之后您应当至少对于Llama系列了如指掌, 但是编写出来的代码通常既不高效也缺乏可维护性. 下面的内容将至少对其中一方面大有裨益.


### 高效Attention

[Flash Attention](https://github.com/Dao-AILab/flash-attention)

Paged Attention: 大名鼎鼎的vllm已经不用自家的核了; 看看远方的[flashinfer](https://github.com/flashinfer-ai/flashinfer)吧

Sparse Attention: 这方面最有名的是xFormers

quiz: 如何使用flash attention推理不等长句子?

### Pytorch Profiler

[profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 将是你在纯python世界的最后一个最常使用到的东西. 养成好习惯: 编写单元测试 + profiler测量.

### torch.compile

[一行代码](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)进入JIT的世界. 不过请注意, dynamic=False的代码通常容易多次触发重编译.

**必要任务**: 使用torch.compile + 静态cache加速你之前的llama代码. 看看能到多快!

### Pytorch Cuda Stream

https://pytorch.org/docs/stable/generated/torch.cuda.stream.html

[臭名昭著](https://github.com/pytorch/pytorch/issues/59692)的Torch Cuda Stream. 了解Cuda的异步执行机制非常重要, 但是Torch提供的Stream效用非常有限.

### Openai Triton

https://triton-lang.org/main/index.html

推荐把tutorial里面的所有核都自己动手写一遍. 

### Nvidia Cuda

> 我真的需要放链接吗 :X

### Cuda Graphs

该操作经常和torch.compile连用. 具体可以参考https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

值得一提的是, Cuda Graph和一些关键的加速技巧冲突, 例如Continous Batching.

