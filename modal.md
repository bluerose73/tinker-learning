# Modal

Modal is a general cloud provider - it is not specially designed for LLM training, although you can use it to train LLMs.

## The Modal Workflow

Using Modal, you can
- define container images
- request nodes, load the image,
- run code on the nodes.


That sounds like traditional serverless computing. What makes Modal different is that all of the steps is done in Python code. Take the example below:

```python
import modal

# Creating a container image
image = modal.Image.debian_slim().pip_install("torch")
app = modal.App(image=image)

# Requesting an A100 node and run a function on it
@app.function(gpu="A100")
def run():
    import torch

    assert torch.cuda.is_available()
```

## Modal for LLM Training

Modal basically gives you a GPU node and let's you create any docker image and run any Python function in it. So, you can train models using popular frameworks, like [unsloth](https://unsloth.ai/).

You just code the entire training code as if you were to run it in an VM.

Note that Modal does not support multi-node training. It is in private test.

[Example LLM fine-tuning using unsloth](https://modal.com/docs/examples/unsloth_finetune)
