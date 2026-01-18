# tinker-learning

[Tinker](https://thinkingmachines.ai/tinker/) is a low-level cloud training API. Writing an SFT job using Tinker API takes 100+ lines.

To make things easier, they build many tools and recipes in [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). That said, the tools in tinker-cookbook is not very well documented. Users often need to hit the code.

In this repo, I will demonstrate how to use tinker for inference, SFT, and RL, using GSM8k dataset as an example. Below are the task definitions.

**Inference:** `inference_gsm8k.py`

Input
- model reference (name of a model on tinker cloud, can be vanilla or fine-tuned)
- jsonl file containing questions

Output
- jsonl file containing questions + answers

**SFT:** `sft_gsm8k.py`

Input
- base model name
- hyper-params
- jsonl file containing questions + answers

Output
- a fine-tuned model reference

**RL:** `rl_gsm8k.py`

Input
- base model name
- hyper-params
- jsonl file
- grader function

Output:
- a fine-tuned model reference

Besides, an eval script `eval_gsm8k.py` takes the inference result and prints a score. It does not use tinker.