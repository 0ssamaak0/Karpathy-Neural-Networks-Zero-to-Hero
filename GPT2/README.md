<div align = "center">
    <img src = "https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/images/cover_GPT2.png?raw=true">

<d>
</div>

# Walkthrough
This folder includes the walkthrough [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t) Video Tutorial and [build-nanogpt](https://github.com/karpathy/build-nanogpt) repo.

This is meant to be a step by step guide to follow along with the video rather than replacing it or reproducing the training process.

To make it easier I divided it into notebooks following the video sections.

<!-- create a markdown table with titles and timestamps -->
| Notebook | Description | Timestamp | Notes|
| --- | --- | --- | --- |
|[1_1_define_arch](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/GPT2/1_1_define_arch.ipynb) | Define the architecture of GPT and run it using 🤗 transformers weights|[0:00](https://www.youtube.com/watch?v=l8pRSuU81PU&t=0s)|-|
|[1_2_dataloader_and_init](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/GPT2/1_2_dataloader_and_init.ipynb) |Prepare the dataset, implement the training loop and fix some bugs|[45:50](https://www.youtube.com/watch?v=l8pRSuU81PU&t=2750s)| - |
|[2_optimization](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/GPT2/2_optimization.ipynb) |Use Tensor Cores, mixed precision, torch.compile, flash attn and change vocab size|[01:22:18](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s)|Run this Notebook with an Nvidia GPU (Ada, Ampere or later)|
|[3_1_hparams](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/GPT2/3_1_hparams.ipynb)|Optimizer parameters, LR Scheduler, Weight decay and batch accumulation|[02:14:55](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s)| - |
|[3_2_DDP](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/GPT2/3_2_DDP.ipynb) |DDP, steps only|[02:46:52](https://www.youtube.com/watch?v=l8pRSuU81PU&t=10012s)|You should also have multiple GPUs to run this code|

The rest of the video is relatively simple and you can just use the code from the original repo.