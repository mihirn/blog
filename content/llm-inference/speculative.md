---
title: "Speculative Decoding"
weight: 6
description: "Speculative decoding, EAGLE, and Medusa Trees"
---

Even when a model fits entirely into HBM and only a single model is being
executed on the GPU, the [primary](https://arxiv.org/abs/2503.08311)
[bottleneck](https://openinfer.io/news/2025-08-05-boosting-local-inference-with-speculative-decoding/)
during LLM inference is the movement of model weights from the HBM, through the
cache hierarchy into device registers for actual computation. This happens
multiple times in a single batch, since different layers are loaded one after
the other as the computation proceeds.

[Speculative
decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
attempts to address this by speculating a number of tokens cheaply, which can
then be verified in parallel by the model. This verification is similar to
prefill: the model is given all the speculated tokens with the appropriate
masking to prevent tokens from being able to see in the future and asked to
generate the entire batch, which is compared with the speculated tokens. The
cost of fetching model weights is effectively amortized over multiple tokens;
the verified tokens are added to the KV cache, while the extra tokens are
discarded.

The traditional way of speculating the tokens is to use a Draft-Target-Model
architecture, where a smaller model acts as a _draft_ model to speculatively
generate tokens, while the actual full-sized model acts as the _target_ which
verifies them in parallel and keeps only the tokens that match. Both the draft
and target models reside in the HBM, so speculative decoding actually increases
the overall memory required during inference. The number of draft tokens to
generate, called the _lookahead window_, depends on the acceptance rate of the
tokens and is empirically determined to balance the potential speedup with
wasted work, with common values being
[3-12](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/).

One option for the draft model is to use a smaller version of the target model,
for instance using Llama-8B as a draft model for a target of Llama-70B.
[EAGLE](https://sites.google.com/view/eagle-llm) uses another approach, where
it trains a lightweight transformer-like decoder which uses the internal
(feature) state of the target model to predict its draft tokens, rather than
just relying on the output tokens. This reduces the overall uncertainty of
predictions and results in a higher acceptance rate.

Another approach to speculative decoding is to eschew the draft model
altogether and augment the original model to generate the draft tokens. In
[Medusa](https://sites.google.com/view/medusa-llm), the original model is
frozen and then a number of heads are added (similar to fine-tuning), where the
$i^{th}$ head is to generate the draft token in the $i^{th}$ position. The
Medusa heads are independent of each other and each generates tokens without
knowing the actual predictions of prior heads, which means a misprediction by
an early head often invalidates the entire draft sequence.

Medusa gets around this by taking the top-$k$ predictions from every head and
then generates a number of possible draft sequences to validate using the
Cartesian product of the individual outputs, called the _Medusa Tree_. Since
the total number of possibilities is exponential in the number of heads and k,
these draft sequences are usually pruned and only the most likely candidates
validated in a single pass. The validated tokens from the candidate with the
highest number of validated tokens are selected and the process continues.
Speculative decoding is an active research topic and there’s a comprehensive
survey of the different techniques
[here.](https://github.com/hemingkx/SpeculativeDecodingPapers?tab=readme-ov-file)

