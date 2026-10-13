<!--
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
-->

[![Formatting](https://github.com/deepspeedai/DeepSpeed-MII/actions/workflows/formatting.yml/badge.svg?branch=main)](https://github.com/deepspeedai/DeepSpeed-MII/actions/workflows/formatting.yml)
[![nv-v100-legacy](https://github.com/deepspeedai/DeepSpeed-MII/actions/workflows/nv-v100-legacy.yml/badge.svg?branch=main)](https://github.com/deepspeedai/DeepSpeed-MII/actions/workflows/nv-v100-legacy.yml)
[![nv-a6000-fastgen](https://github.com/deepspeedai/DeepSpeed-MII/actions/workflows/nv-a6000-fastgen.yml/badge.svg?branch=main)](https://github.com/deepspeedai/DeepSpeed-MII/actions/workflows/nv-a6000-fastgen.yml)
[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/deepspeedai/DeepSpeed/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepspeed-mii.svg)](https://pypi.org/project/deepspeed-mii/)

<div align="center">
 <img src="docs/images/mii-white.svg#gh-light-mode-only" width="400px">
 <img src="docs/images/mii-dark.svg#gh-dark-mode-only" width="400px">
</div>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 最新动态

* [2024/01] [DeepSpeed-FastGen: 推出对 Mixtral、Phi-2 和 Falcon 的支持，带来重大的性能与功能升级。](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19)
* [2023/11] [DeepSpeed-FastGen: 基于 MII 与 DeepSpeed-Inference 实现 LLM 的高吞吐文本生成](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
* [2022/11] [基于 DeepSpeed MII 在 1 秒内完成 Stable Diffusion 图像生成](mii/legacy/examples/benchmark/txt2img)
* [2022/10] [发布 DeepSpeed 模型推理实现库 (MII)](https://www.deepspeed.ai/2022/10/10/mii.html)

# 目录

<!-- toc -->

- [DeepSpeed-MII](#deepspeed-mii)
- [核心关键技术](#核心关键技术)
- [MII 工作原理](#mii-工作原理)
- [支持的模型架构](#支持的模型架构)
- [快速入门](#快速入门)
- [参与贡献](#参与贡献)
- [商标声明](#商标声明)

<!-- tocstop -->

# DeepSpeed 模型推理实现库 (MII) <a name="deepspeed-mii"></a>

隆重推出 MII —— 一款由 DeepSpeed 设计的开源 Python 工具库，旨在普及强大的大模型推理能力，专注提供**高吞吐量**、**低延迟**与**高性价比**。

* MII 具备分块 KV 缓存 (Blocked KV-caching)、连续批处理 (Continuous Batching)、动态 SplitFuse (Dynamic SplitFuse)、张量并行 (Tensor Parallelism) 以及高性能 CUDA 算子等核心特性，能够为 Llama-2-70B、Mixtral (MoE) 8x7B 和 Phi-2 等主流 LLM 提供极致的快速高吞吐文本生成。v0.2 版本的最新更新增添了更多模型系列支持、性能优化与功能强化。相比 vLLM 等业界主流系统，MII 能够提供高达 2.5 倍的有效吞吐量。关于详细性能基准测试结果，请参阅[最新 DeepSpeed-FastGen 博客](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19)与 [DeepSpeed-FastGen 发布博文](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen)。

<div align="center">
 <img src="docs/images/fastgen-24-01-hero-light.png#gh-light-mode-only" width="850px">
 <img src="docs/images/fastgen-24-01-hero-dark.png#gh-dark-mode-only" width="850px">
</div>

<div align="center">
 <img src="docs/images/fastgen-hero-light.png#gh-light-mode-only" width="800px">
 <img src="docs/images/fastgen-hero-dark.png#gh-dark-mode-only" width="800px">
</div>

* 我们于 2022 年[首次发布 MII](https://www.deepspeed.ai/2022/10/10/mii.html)，涵盖了 v0.0.9 及之前的所有历史版本。除了语言模型外，我们还支持加速 [Stable Diffusion 等文生图模型 (text2image)](mii/legacy/examples/benchmark/txt2img)。有关更早版本的详细信息，请参阅我们的[传统 Legacy API](mii/legacy/)。

# 核心关键技术

## 适用于高吞吐文本生成的 MII

MII 通过采用以下四项核心技术实现加速文本生成推理：

* **分块 KV 缓存 (Blocked KV Caching)**
* **连续批处理 (Continuous Batching)**
* **动态 SplitFuse (Dynamic SplitFuse)**
* **高性能 CUDA 算子 (High Performance CUDA Kernels)**

如需深入了解这些特性的技术细节，请[参考官方技术博客](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen)，文中还包含了详尽的性能分析评测。

## MII 传统技术特性 (Legacy)

在过去的版本中，MII 为低延迟模型服务场景引入了数项[关键性能优化](https://www.deepspeed.ai/2022/10/10/mii.html#inference-optimizations-with-mii)：

* 面向 Transformer 模型的 DeepFusion 融合优化
* 基于张量切片 (Tensor-Slicing) 的多 GPU 推理
* 面向资源受限系统的 ZeRO-Inference 推理优化
* 深度编译器级优化

# MII 工作原理

<div align="center">
 <img src="docs/images/mii-arch-light.png#gh-light-mode-only" width="800px">
 <img src="docs/images/mii-arch-dark.png#gh-dark-mode-only" width="800px">
</div>

图 1：MII 架构图，展示了 MII 在部署开源模型前如何利用 DS-Inference 自动完成针对性优化。图中展示的 DeepSpeed-FastGen 优化技术详情已发布于[官方博文](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen)。

在底层，MII 由 [DeepSpeed-Inference](https://github.com/deepspeedai/DeepSpeed) 强力驱动。根据模型架构、模型参数规模、批处理大小以及可用硬件资源，MII 会自动应用相应的一整套系统级优化，以最大限度降低推理延迟并最大化系统吞吐量。

# 支持的模型架构

MII 目前支持横跨八大主流模型架构的 37,000 余个开源模型。我们计划在近期持续扩展更多模型，如果您希望支持特定的模型架构，欢迎[提交 Issue](https://github.com/deepspeedai/DeepSpeed-MII/issues) 告知我们。当前所有模型均在后端利用 Hugging Face Hub 获取模型权重及对应的分词器 (Tokenizer)。当前版本原生支持以下模型系列：

模型系列 | 参数规模范围 | 约模型数量
------ | ------ | ------
[Falcon](https://huggingface.co/models?other=falcon) | 7B - 180B | 600
[Llama](https://huggingface.co/models?other=llama) | 7B - 65B | 57,000
[Llama-2](https://huggingface.co/models?other=llama-2) | 7B - 70B | 1,200
[Llama-3](https://huggingface.co/models?other=llama-3) | 8B - 405B | 1,600
[Mistral](https://huggingface.co/models?other=mistral) | 7B | 23,000
[Mixtral (MoE)](https://huggingface.co/models?other=mixtral) | 8x7B | 2,900
[OPT](https://huggingface.co/models?other=opt) | 0.1B - 66B | 2,200
[Phi-2](https://huggingface.co/models?other=phi) | 2.7B | 1,500
[Qwen](https://huggingface.co/models?other=qwen) | 7B - 72B | 500
[Qwen2](https://huggingface.co/models?other=qwen2) | 0.5B - 72B | 3,700

## MII 传统支持模型 (Legacy)

MII Legacy API 支持超过 50,000 个不同模型，包括 BERT、RoBERTa、Stable Diffusion 以及 Bloom、GPT-J 等文本生成模型。有关完整列表，请参见我们的[传统支持模型与任务表格](mii/legacy/#supported-models-and-tasks)。

# 快速入门

DeepSpeed-MII 允许用户仅用数行代码即可为受支持的模型搭建非持久化流水线或持久化服务部署。

- [安装指南](#安装指南)
- [非持久化流水线](#非持久化流水线)
- [持久化服务部署](#持久化服务部署)

## 安装指南

最快的使用方式是通过 [DeepSpeed-MII 的 PyPI 发布版本](https://pypi.org/project/deepspeed-mii/) 进行安装，只需几分钟即可准备就绪：

```bash
pip install deepspeed-mii
```

为了提升易用性并显著减少该领域诸多项目所必需的冗长编译等待时间，我们通过名为 [DeepSpeed-Kernels](https://github.com/deepspeedai/DeepSpeed-Kernels) 的独立代码库分发了预编译的 Python Wheel 包，覆盖了绝大部分自定义内核算子。在具备计算能力 8.0+（Ampere 架构及以上）、CUDA 11.6+ 和 Ubuntu 20+ 的 NVIDIA GPU 环境下，该库具备极佳的通用便携性。在绝大多数情况下，您无需显式感知该库的存在，它作为 DeepSpeed-MII 的前置依赖会自动安装。但是，如果您由于特定原因需要手动编译内核算子，请参阅我们的[进阶编译安装文档](https://github.com/deepspeedai/DeepSpeed-Kernels#source)。

## 非持久化流水线

非持久化流水线 (Non-Persistent Pipeline) 是体验 DeepSpeed-MII 的极佳方式。非持久化流水线仅在当前运行的 Python 脚本生命周期内有效。部署运行非持久化流水线的完整示例仅需 4 行代码，不妨亲自一试：

```python
import mii
pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)
```

返回的 `response` 是一个 `Response` 对象列表。我们可以获取有关生成的各项详细信息（例如 `response[0].prompt_length`）：

- `generated_text: str` 模型生成的文本内容。
- `prompt_length: int` 原始提示词中的 Token 数量。
- `generated_length: int` 模型生成的 Token 数量。
- `finish_reason: str` 停止生成的终止原因。`stop` 表示生成了结束符 (EOS)，`length` 表示达到了 `max_new_tokens` 或 `max_length` 上限。

如果您希望释放显存并销毁流水线，请调用 `destroy` 方法：

```python
pipe.destroy()
```

### 张量并行 (Tensor Parallelism)

利用 MII 可以轻松借助多 GPU 系统获得更高性能。当使用 `deepspeed` 启动器运行时，张量并行度由 `--num_gpus` 参数自动控制：

```bash
# 在单 GPU 上运行
deepspeed --num_gpus 1 mii-example.py

# 在多 GPU 上运行
deepspeed --num_gpus 2 mii-example.py
```

### 流水线配置选项

虽然搭建非持久化流水线部署仅需传入模型名称或路径，但我们为用户提供了丰富的定制选项：

**`mii.pipeline()` 配置项**：
- `model_name_or_path: str` [HuggingFace](https://huggingface.co/) 上的模型名称或本地路径。
- `max_length: int` 设置 prompt + response 的默认最大 Token 长度。
- `all_rank_output: bool` 开启后，所有 rank 都会返回生成的文本。默认情况下仅 rank 0 返回文本。

用户还可以在调用 `pipe()` 时针对各个单独提示词控制文本生成特征：

- `max_length: int` 设置针对该 prompt 的 prompt + response 最大 Token 长度。
- `min_new_tokens: int` 设置生成的最小 Token 数量，`max_length` 优先级高于此设置。
- `max_new_tokens: int` 设置生成的最大 Token 数量。
- `ignore_eos: bool`（默认为 `False`）设置为 `True` 可防止在遇到 EOS token 时提前终止生成。
- `top_p: float`（默认为 `0.9`）设置为小于 `1.0` 时过滤 token，仅保留累积概率和 &ge; `top_p` 的最高概率词元。
- `top_k: int`（默认为 `None`）为 `None` 时禁用 top-k 过滤；设置数值时保留最高概率的 k 个词元。
- `temperature: float`（默认为 `None`）为 `None` 时禁用采样温度调节；设置时用于调整词元概率分布的平滑度。
- `do_sample: bool`（默认为 `True`）设置为 `True` 时从输出 logits 进行随机采样；设置为 `False` 时采用贪心解码 (greedy search)。
- `return_full_text: bool`（默认为 `False`）设置为 `True` 时在返回结果开头拼接原始输入提示词。

## 持久化服务部署

持久化部署 (Persistent Deployment) 非常适合长时间运行与生产环境应用程序。持久化模型通过轻量级 gRPC 服务运行，支持多个客户端并发请求。运行持久化模型的完整示例仅需 5 行代码，不妨亲自一试：

```python
import mii
client = mii.serve("mistralai/Mistral-7B-v0.1")
response = client.generate(["Deepspeed is", "Seattle is"], max_new_tokens=128)
print(response)
```

返回的 `response` 同样为 `Response` 对象列表，可访问其各项生成属性（如 `response[0].prompt_length`）：

- `generated_text: str` 模型生成的文本内容。
- `prompt_length: int` 原始提示词中的 Token 数量。
- `generated_length: int` 模型生成的 Token 数量。
- `finish_reason: str` 停止生成的终止原因（`stop` 或 `length`）。

如果您希望从其他独立进程发起生成请求，也可以轻松连接：

```python
client = mii.client("mistralai/Mistral-7B-v0.1")
response = client.generate("Deepspeed is", max_new_tokens=128)
```

当不再需要持久化服务时，可以通过任意客户端关闭服务器进程：

```python
client.terminate_server()
```

### 模型并行 (Model Parallelism)

利用多 GPU 系统提升服务的吞吐量与降低延迟同样非常简单。模型并行由传递给 `mii.serve` 的 `tensor_parallel` 参数控制：

```python
client = mii.serve("mistralai/Mistral-7B-v0.1", tensor_parallel=2)
```

该配置将把模型切分到 2 张 GPU 上协同运行，相较于单 GPU 能够实现更低的推理延迟与更高的吞吐量。

### 模型副本负载均衡 (Model Replicas)

我们还可以通过配置多个模型副本 (Model Replicas) 来充分利用多 GPU（乃至多节点）系统，并享受 DeepSpeed-MII 自带的请求负载均衡机制：

```python
client = mii.serve("mistralai/Mistral-7B-v0.1", replica_num=2)
```

上述部署将启动 2 个模型副本（每张 GPU 一个），并在两个模型实例之间自动对入站请求进行负载均衡。

模型并行与模型副本亦可组合使用，以应对更大规模的多 GPU 集群。在以下示例中，我们在 4 GPU 系统上运行 2 个模型副本，每个副本横跨 2 张 GPU 进行张量并行切分：

```python
client = mii.serve("mistralai/Mistral-7B-v0.1", tensor_parallel=2, replica_num=2)
```

选择采用模型并行还是模型副本以获取最佳性能，取决于具体硬件配置、模型架构以及工作负载特征。例如，对于较小的模型，使用模型副本往往能为并发请求带来更低的平均延迟；而对于大型模型，仅使用模型并行通常能够获得更高的整体系统吞吐量。

### RESTful API 服务

在创建持久化 MII 部署时，通过设置 `enable_restful_api=True`，可以非常便捷地对外暴露 RESTful API 接口。RESTful API 服务端点格式为 `http://{HOST}:{RESTFUL_API_PORT}/mii/{DEPLOYMENT_NAME}`。完整示例如下：

```python
client = mii.serve(
    "mistralai/Mistral-7B-v0.1",
    deployment_name="mistral-deployment",
    enable_restful_api=True,
    restful_api_port=28080,
)
```

---
📌 **说明：** 显式指定 `deployment_name` 并非强制要求（MII 会自动为您生成），但显式命名是最佳实践，有助于确保与正确的 RESTful API 路径进行交互。

---

配置完成后，您可以使用任何 HTTP 客户端向 RESTful 网关发送提示词请求，例如使用 `curl`：

```bash
curl --header "Content-Type: application/json" --request POST  -d '{"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}' http://localhost:28080/mii/mistral-deployment
```

或者使用 `python` 中的 `requests` 库：

```python
import json
import requests
url = f"http://localhost:28080/mii/mistral-deployment"
params = {"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}
json_params = json.dumps(params)
output = requests.post(
    url, data=json_params, headers={"Content-Type": "application/json"}
)
```

### 持久化部署配置选项

尽管搭建持久化部署只需提供模型名称或路径，我们同样为用户提供了丰富的定制选项。

**`mii.serve()` 配置项**：
- `model_name_or_path: str`（必需）[HuggingFace](https://huggingface.co/) 模型名称或本地路径。
- `max_length: int`（默认为模型配置中的最大上下文序列长度）设置 prompt + response 的默认最大 Token 长度。
- `deployment_name: str`（默认为 `f"{model_name_or_path}-mii-deployment"`）持久化模型的唯一标识符字符串。若显式指定，后续客户端应通过 `client = mii.client(deployment_name)` 进行获取。
- `tensor_parallel: int`（默认为 `1`）模型张量并行切分的 GPU 卡数。
- `replica_num: int`（默认为 `1`）启动的模型副本数量。
- `enable_restful_api: bool`（默认为 `False`）开启后将启动一个 RESTful API 网关进程，可通过 `http://{host}:{restful_api_port}/mii/{deployment_name}` 访问。详见 [RESTful API 章节](#restful-api-服务)。
- `restful_api_port: int`（默认为 `28080`）当 `enable_restful_api` 为 `True` 时，用于与 RESTful API 交互的端口号。

**`mii.client()` 配置项**：
- `model_or_deployment_name: str` 传递给 `mii.serve()` 的模型名称或 `deployment_name`。

用户在调用 `client.generate()` 时，同样可以使用以下参数控制单次生成的特征：

- `max_length: int` 设置针对该 prompt 的 prompt + response 最大 Token 长度。
- `min_new_tokens: int` 设置生成的最小 Token 数量，`max_length` 优先级高于此设置。
- `max_new_tokens: int` 设置生成的最大 Token 数量。
- `ignore_eos: bool`（默认为 `False`）设置为 `True` 可防止在遇到 EOS token 时提前终止生成。
- `top_p: float`（默认为 `0.9`）设置为小于 `1.0` 时过滤 token，仅保留累积概率和 &ge; `top_p` 的最高概率词元。
- `top_k: int`（默认为 `None`）为 `None` 时禁用 top-k 过滤；设置数值时保留最高概率的 k 个词元。
- `temperature: float`（默认为 `None`）为 `None` 时禁用采样温度调节；设置时用于调整词元概率分布的平滑度。
- `do_sample: bool`（默认为 `True`）设置为 `True` 时从输出 logits 进行随机采样；设置为 `False` 时采用贪心解码。
- `return_full_text: bool`（默认为 `False`）设置为 `True` 时在返回结果开头拼接原始输入提示词。

# 参与贡献

本项目欢迎广大开发者提交贡献与改进建议。

DeepSpeed-MII 已全面采纳[开发者原创性声明 (DCO)](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin)。所有 deepspeedai 旗下开源仓库均强制要求 DCO。
（DeepSpeed 此前曾使用 CLA，现已全面迁移至 DCO）。

在 Git 提交信息中附带 `Signed-off-by:` 行即可提供 DCO 签名。在运行 `git commit` 时添加 `-s` 参数会自动追加该签名行。
例如，执行 `git commit -s -m 'commit info.'` 将生成包含如下信息的提交：`commit info. Signed-off-by: My Name <my_email@my_company.com>`。
GitHub 上的 DCO 机器人会确保提交签名中的电子邮箱地址与提交作者严格一致，满足条件后 PR 即可进入合并审查流程。

本项目已采纳 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
如需了解更多信息，请参阅[行为准则常见问题解答](https://opensource.microsoft.com/codeofconduct/faq/)，如有任何其他疑问或反馈，请联系 [opencode@microsoft.com](mailto:opencode@microsoft.com)。

# 商标声明

本项目可能包含针对某些项目、产品或服务的商标或徽标。经授权使用 Microsoft 商标或徽标必须严格遵守并遵循 [Microsoft 商标与品牌指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。
在修改本项目的衍生版本中使用 Microsoft 商标或徽标不得引起混淆或暗示 Microsoft 官方赞助。
任何第三方商标或徽标的使用均须遵循该第三方的相关规定。

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月13日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
