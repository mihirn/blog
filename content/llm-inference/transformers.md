---
title: "LLMs and Transformers"
weight: 1
description: "Introduction, embeddings, transformers and attention mechanisms"
---

Large Language Models (LLMs) are neural networks that take in a prompt (a
sequence of words) and continuously generate the next word in the sequence
until some termination condition is reached. The input prompt is converted to a
set of vectors using an embedding matrix; these vectors are then sent through a
[Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
(a kind of neural network), which outputs a probability distribution across the
vocabulary of the language, where each entry represents a potential next word
in the sequence.

One of these potential entries is selected (often the highest probability one,
but with some randomness using a temperature setting to add some
non-determinism), which is then converted back to a word using an output
matrix, which logically serves as the reverse of the embedding matrix. This
process continues iteratively, typically either until a specially designated
end-of-sequence (EOS) word is chosen as the next one or a maximum response
length is hit.

## Embeddings

In reality, LLMs work on tokens, which are usually subsets of words, rather
than entire words. The embedding matrix is a static conversion from these
tokens to a vector in an n-dimensional embedding space, with the intuition that
similar words are closer together in that space. The dimensionality of the
model is a fixed hyperparameter determined before the model is trained and is
represented by $d_{model}$.

Directions (or operations) in this space represent _ideas_; for example, the
idea of an _important playwright_ could be embedded in the space between the
vectors representing Elizabethan England and Shakespeare. When the same
operation is applied to ancient Greece, one might get Sophocles.

## Transformers

Transformers are a neural network architecture with a large number of layers,
each of which has primarily two phases: the _attention_ phase and the
_feed-forward network_ phase, often separated by normalization operations. The
embedding vectors of the input form the initial context; this context is then
modified by each of these layers using giant matrix multiplication operations.

The basic idea around attention is that the embeddings from the input text are,
during inference, static isolated conversions and lack any knowledge about the
relationships between the tokens. In the attention phase, all the vectors in
the context operate and modify each other: this allows different connotations
of the same word to be teased apart. As an example, the word “bear” in “a brown
bear” and “doesn’t bear any fruit” start off with the same embedding but after
several attention phases represent different points in the embedding space.

The module of the Transformer responsible for attention is called the
_attention head_. Each layer has multiple attention heads, and takes a matrix
representing the embeddings of the context and outputs an identically-sized
matrix which forms the context that is then sent to the feed-forward layer and
eventually to the next layer of the model. The matrix operations are divided
evenly amongst the heads of the layer, with each one producing partial results
which are concatenated to form the entire context for the feed-forward layer.

Each attention head operates using three parameters—Query (Q), Key (K), and
Value (V)—which are derived from the input tokens. It has three model
parameters (the $W^Q$, $W^K$, and $W^V$ matrices) which are static parts of the
model learnt during the training process. There is also a fourth parameter,
$W^O$, which is common across all the heads of a layer. The head has an
associated dimensionality, represented by $d_k$, which is another fixed
hyperparameter determined prior to training (similar to the dimensionality of
the model). The size of these matrices is determined by both these
dimensionalities making them ($d_{model} \times d_k$). Typically, the value
$d_k$ is chosen so that all the heads in a layer have the same dimensionality
as the model ($d_{model}$), i.e., if there is a single head $d_k = d_{model}$,
but if there are $h$ heads then $d_k = d_{model}/h$. The number of heads is
also a fixed hyperparameter of the model and cannot trivially be changed after
the model is trained.

Q, K, and V are calculated by multiplying the vector representing the input
tokens, along with embedded positional information, with $W^Q$, $W^K$, and
$W^V$. While all three matrices operate on the same input tokens, the weight
matrices represent different bits of _knowledge_ learnt by the model during
training and transform and project the input tokens in different ways.

Assuming an input sequence length of $k$ tokens, the context matrix is a ($k
\times d_{model}$) matrix with each of the rows representing the embeddings of
a single token. Each of the $k$ input token vectors is ($1 \times d_{model}$),
and the weight matrices are ($d_{model} \times d_k$), so the resulting Q and K
vectors are ($1 \times d_k$). Combining $k$ such vectors (one for each token)
results in ($k \times d_k$) sized Q and K matrices. Multiplying the Q and
transposed K matrices ($K^T$) gives us a ($k \times k$) sequence-length square
sized matrix. The values of this matrix grow in proportion to $d_k$ and can
grow very large, so they are divided by $\sqrt{d_k}$ and then normalized using
a softmax function which converts them into a probability distribution called
the _attention scores_.

The attention score matrix has a row and a column for each token in the context
and represents the effect that each token has on every other token in the
context or how much a token is influenced by another token. In ML parlance,
this is how much a token attends to another token: $A_{i,j}$ is the amount that
token $i$ of the context attends to token $j$. Attention scores are not
symmetric and, in some cases, are deliberately masked out to prevent later
tokens from having an undue influence on earlier ones, particularly while
generating outputs.

The attention scores are then multiplied by the Value (V) matrix to get the
updated embedding values for every token in the context. The embedding values
from all the heads are concatenated and then multiplied by the $W^O$ matrix, so
that the independent knowledge (or perspectives) learnt from each head can be
mixed into a single, coherent representation in the updated context matrix.
This matrix is then passed through a feed-forward neural network, i.e., a
fully-connected neural network without any loops or cross-token dependencies,
so token processing is fully parallelizable. This output can then be passed to
the next transformer layer.

Attention performed using multiple heads, or _multi-head attention (MHA)_
allows the model to learn multiple relationships between tokens, which are
often lost in the single-head case; for example, one head could learn the
cardinality relationships between words, while another might represent their
temporal relationship. This is not possible with a single head because the
attention scores are calculated as probabilities and a single head is only able
to represent the most prominent relationships between the tokens.

In single and multi-head attention, each attention head has its own $W^Q$,
$W^K$, and $W^V$ matrices and consequently different Q, K, and V matrices from
the same input tokens, which has a significant memory footprint. [Multi-query
attention](https://arxiv.org/pdf/1911.02150) (MQA) reduces this footprint by
having only a single pair of $W^K$ and $W^V$ matrices which are shared by all
the heads (each head still maintains its own $W^Q$ matrix); consequently, the K
and V matrices can be reused across all the heads, which reduces the memory
footprint by a factor of $h$, at the cost of quality degradation.

[Grouped-query attention](https://arxiv.org/pdf/2305.13245v2) (GQA) is an
intermediate point in the design space where the heads are divided into $g$
groups, each of which share the same $W^K$ and $W^V$ matrices. GQA is flexible
and tunable: if $g = h$, GQA is the same as MHA, while if $g = 1$, GQA is the
same as MQA. GQA is popular because it offers model quality comparable to MHA
while still reducing the memory footprint by a factor of $h/g$. Models also do
not have to be re-trained from scratch to be deployed using GQA; the heads from
an MHA model can be combined using _mean pooling_ and _uptrained_ using a tiny
fraction of compute required for the original training (\~5%).

Another approach to preserve the expressive power of MHA with the memory
footprint reduction of MQA and GQA is [Multi-head Latent
Attention](https://planetbanatt.net/articles/mla.html) (MLA), introduced by
[DeepSeek-V2](https://arxiv.org/pdf/2405.04434). In the former, every head has
its own $W^K$ and $W^V$ matrices, which are multiplied by the input tokens to
get the K and V matrices, while in the latter multiple heads share the $W^K$
and $W^V$ matrices. MLA splits each of the $W^K$ and $W^V$ matrices into a pair
of smaller matrices, which can be combined to form matrices of the original
size. This is called low-rank approximation, where a ($d_{model} \times d_k$)
matrix can be replaced by the product of a ($d_{model} \times d_{latent}$) and
a ($d_{latent} \times d_k$) matrix, where $d_{latent} \ll d_k$. These are
sometimes called the down and the up weights or $W^D$ and $W^U$, since they
compress and decompress the data.

In MLA, every head shares the down matrices, i.e., a single layer has only one
$W^{KD}$ and one $W^{VD}$ matrix which is shared; every input token is
multiplied by these matrices and the resulting representation can be shared by
all the heads in the layer. However, every head performs an additional
multiplication with the $W^{KU}$ and $W^{VU}$ matrices before attention,
resulting in unique K and V vectors, so every head gets its own unique
interpretation of the data, similar to MHA. Intuitively, it is like a
compressed version of the token’s vector is shared across all the heads, but
each head then decompresses the token with its own parameterization. This
decompression can be fused with existing attention steps so that there are no
unnecessary additional matrix multiplications.

The dimensionality of the latent matrices ($d_{latent}$) is a fixed
hyperparameter, but the matrices themselves are learnt during the training
process. While models traditionally had to be retrained from scratch to use
MLA, there is [active](https://github.com/JT-Ushio/MHA2MLA)
[research](https://github.com/MuLabPKU/TransMLA) into methods that allow MHA
and GQA models to be converted to MLA with little fine tuning. MLA is popular
because it offers quality comparable to MHA (and sometimes even superior) with
a memory footprint comparable to GQA.

## Mixture-of-Experts

An extension of the traditional Transformer architecture is a
[_Mixture-of-Experts_](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
([_MoE_](https://huggingface.co/blog/moe)) model. Every layer of a traditional
Transformer has an attention phase and a dense, fully-connected feed-forward
network. In MoE models, the attention phase of every layer remains the same,
but then rather than having a single, large feed-forward network, there are a
number of smaller feed-forward networks (each called an expert) and a routing
layer which directs the tokens to only a subset of the experts. As an example,
a dense 64B parameter feed-forward network could be replaced by eight 8B
parameter experts; while the total number of parameters remains the same,
unlike the dense network, only a subset of them are activated per token.

The number and size of the experts, as well as the number of experts each token
is sent to, are hyperparameters determined prior to training the model. A
common routing model is to send tokens to the top-$k$ experts and then combine
the outputs using a weighted sum. Tokens do not have to select the same experts
for every layer they pass through. The routing layer is a critical part of MoE
models: if all the requests are sent to the same expert, it is effectively a
traditional Transformer, except with a much smaller feed-forward network, and
the quality of the model suffers. Further, even a single undertrained expert
impacts model quality.

The routing layer is thus responsible for ensuring that tokens are evenly
distributed across experts and that individual experts are adequately trained,
and is itself a model which is learnt during training. It uses a number of
techniques to load balance across the experts: noise can be added to expert
routing scores to ensure that tokens are evenly distributed, as well as
minimizing the _auxiliary loss_ (or _load balancing loss_), where a high
auxiliary loss represents a non-uniform distribution of tokens across experts.

There is a very brief overview on how a transformer is trained into a usable
model at {{<pagelink "llm-inference/training" >}}, while the discussion about
inference and serving requests continues at {{<pagelink "llm-inference/kvcache"
>}}.

## Additional References

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
1. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
1. [Attention in transformer (Video)](https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en)
1. [Transformer Architecture Explained (Video)](https://www.youtube.com/watch?v=6vThlsJ_ASE)
1. [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
1. [How Attention Got So Efficient (Video)](https://www.youtube.com/watch?v=Y-o545eYjXM)
1. [How DeepSeek Rewrote the Transformer \[MLA\] (Video)](https://www.youtube.com/watch?v=0VLAoVGf_74)
