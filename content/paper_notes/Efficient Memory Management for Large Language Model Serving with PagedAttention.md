---
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
date: 2024-07-05
draft: False
---

# Abstract
The inefficiency management of Key-Value cached memory for each requests will cost a large waste by fragmentation and reduntant duplication of the memory which will lower the efficiency of LLM model serving process by the batch size limitation.

They purpose the **PageAttention** mechanism inspired by the virtual memory management in OS. The LLM serving system (vLLM) achieved:
(1) near-zero waste in KV cache memory
(2) flexible sharing of KV cache within and across requests to further reduce mem- ory usage

Their work improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) and [Orca](https://www.usenix.org/conference/osdi22/presentation/yu).

# Problem Definition and Challenges

# Methodology

# Experiment Design and Result