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

## Rope

该任务将Rope操作从Attention解耦, 允许更灵活的Cache策略.

检查`tasks/rope`获取更多信息。