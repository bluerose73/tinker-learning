# tinker-learning

[Tinker](https://thinkingmachines.ai/tinker/) is a low-level cloud training API. Writing an SFT job using Tinker API takes 100+ lines.

To make things easier, they build many tools and recipes in [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). That said, the tools in tinker-cookbook is not very well documented. Users often need to hit the code.

In this repo, I will demonstrate how to use tinker for inference, SFT, and RL, using GSM8k dataset as an example.

## My comments on Tinker

### Tinker for inference

- Create a `sampling_client` from a model checkpoint
- Run `sampling_client.sample(prompt)` to generate response.

Inference is fast based on my experience. Seems no rate limit.

### Tinker for SFT

- Create a `training_client` from a base model (configure lora rank here)
- User define a custom loss function, or select an existing loss function tinker provides
- Call `training_client.forward_backword(data, loss_fn)` to calculate gradients and the loss. Loss function is passed in as a callback
- Call `optim_step()` to update the weights. It uses Adam optimizer

A [training script](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_basic.py) is available in `tinker-cookbook`.

If the user instead wants to write his own training loop, an example is [here](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py).

### Tinker for RL

There is no specific APIs for RL. Users write RL script by combining inference and SFT APIs

- For each datapoint, sample x rollouts using `sampling_client.sample(prompt)`
- Run grader **locally** to get reward
- Call `training_client.forward_backword(data, loss_fn)` to calculate gradients (`data` passed to `forward_backward` includes reward)
- Create a new sampler from updated weights

An [RL script](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_basic.py) is available in `tinker-cookbook`. Using that script, user only needs to write the dataset class and the grader class (named `Env`).

If the user instead wants to write his own training loop, an example is [here](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_loop.py)

### Tinker features

As is listed above, tinker only provides sampling, forward_backward, and weight update APIs.
It's like writing a training loop using pytorch, but all the GPU operations is now done on tinker cloud.
User needs to implement
- Data loading
- Tokenization
- Logging

But `tinker-cookbook` does this for users.

### Pros and Cons

Pros:
- Users don't need to care about infrastructure - clusters, nodes, gpus, parallel training, etc.
- Medium-High flexibility. Users still have control over the training loop.
- Good throughput

Cons:
- Only supports LoRA with Adam optimizer
- Limited base model selection (~20 open-source models)
- RL grader runs locally. If the grader is heavy, like agent training, users still needs grader infra.

## Scripts in this repo

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
- no jsonl file (dataset is loaded from internet)
- grader function

Output:
- a fine-tuned model reference

Besides, an eval script `eval_gsm8k.py` takes the inference result and prints a score. It does not use tinker.
