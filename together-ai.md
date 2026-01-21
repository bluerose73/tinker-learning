# Model Training Using together.ai

Together.ai provides two services for model training: 1) fine-tuning job and 2) clusters.

## Fine-Tuning Job

Fine-tuning job is a high-level API. User submit a dataset, and the rest is handled by the platform.

User can configure hyper-parameters like learning rate.

Supports LoRA and full fine-tuning. For RL, it only supports Preference Fine-Tuning. Custom grader is not supported.

Example:
```python
client.fine_tuning.create(
    training_file=train_file_resp.id,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    train_on_inputs="auto",
    n_epochs=3,
    n_checkpoints=1,
    wandb_api_key=WANDB_API_KEY,  # Optional, for visualization
    lora=True,  # Default True
    warmup_ratio=0,
    learning_rate=1e-5,
    suffix="test1_8b",
)
```

## Clusters

Users can rent a k8s / slurm cluster. Users have full control over the cluster, they can install their own ML framework and run any kind of training jobs.

The experience should be similar to Azure Kubernetes Service AKS.