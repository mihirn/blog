---
title: "LLM Inference Notes"
description: "LLM inference infrastructure for a systems audience."
---

# LLM Inference Infrastructure

This is an opinionated discussion of the basics of LLM inference and the
ecosystem of serving runtimes powering inference today, from a systems design
perspective. It was originally written internally at [IOP
Systems](https://iop.systems/), but thanks to some gentle ~~arm-twisting~~
persuasion, it has been posted in case it is useful to anyone. (Thanks for the
push, [Yao](https://yaoyue.org/)!)

The intended audience is systems and infrastructure developers who want to
understand the systems behind LLM inference and serving those requests at
scale, but have little to no machine learning knowledge. It focuses largely on
the performance aspects of these systems, specifically around techniques to
scale serving, reduce latency, or drive better efficiency and hardware
utilization.

This is not intended to be comprehensive or complete and favours a broad
intuition over mathematical rigour. That said, it is best thought of as a
living document and corrections and contributions to improve clarity or scope
are welcome [here](https://github.com/mihirn/blog).

[Start reading →]({{< relref "llm-inference/_index.md" >}})
