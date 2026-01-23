# SkyRL

[SkyRL](https://skyrl.readthedocs.io/en/latest/index.html) is an open-source RL framework. It runs RL on a ray cluster.

![system overview](assets/system-overview.webp)

Trainer and generator can share GPUs or not.

## Launching a Job

Below is a GSM8K example. [link](https://skyrl.readthedocs.io/en/latest/getting-started/quickstart.html)

```bash
uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
   # Data setup
   data.train_data="['$HOME/data/gsm8k/train.parquet']" \
   data.val_data="['$HOME/data/gsm8k/validation.parquet']" \

   # Trainer and training algorithm
   trainer.algorithm.advantage_estimator="grpo" \
   trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \

   # Model placement and training strategy (colocate or disaggregate, sharding, etc.)
   trainer.strategy=fsdp2 \
   trainer.placement.colocate_all=true \
   trainer.placement.policy_num_gpus_per_node=4 \

   # Evaluation and checkpointing
   trainer.eval_batch_size=1024 \
   trainer.eval_before_train=true \
   trainer.eval_interval=5 \
   trainer.ckpt_interval=10 \

   # Generator setup for spinning up InferenceEngines
   generator.backend=vllm \
   generator.num_inference_engines=4 \
   generator.inference_engine_tensor_parallel_size=1 \
   generator.weight_sync_backend=nccl \

   # Environment class for the dataset
   # Can be specified here to apply to the full dataset, or at the per-prompt level during preprocessing
   environment.env_class=gsm8k \

   # WandB logging
   trainer.logger="wandb" \
   trainer.project_name="gsm8k" \
   trainer.run_name="gsm8k_test" \

   ... # Other parameters (see `examples/gsm8k/run_gsm8k.sh` for more)
```

## Dataset Format

Below is an example for a "multiply" task, e.g. `2 * 3 = 6`. [link](https://skyrl.readthedocs.io/en/latest/tutorials/new_env.html#)

```python
for idx in range(num_examples):
     question, answer = generate_multiplication_problem(num_digits)

     data = {
         "data_source": "synthetic_multiply",
         "prompt": [
             system_prompt,
             {
                 "role": "user",
                 "content": question,
             }
         ],
         "env_class": "multiply",
         "reward_spec": {
             "method": "rule",
             "ground_truth": answer,
         },
         "extra_info": {
             "num_digits": num_digits,
             "split": split_name,
         },
     }
     examples.append(data)
```

Later, we can access the data from the grader - called Environment.

## Environment

User needs to write a grader.

We subclass `BaseTextEnv` and override the `step` method. It takes model output and returns a reward. 

```python
class MultiplyEnv(BaseTextEnv):
   def _parse_action(self, action: str) -> str:
      """Extract answer from \\boxed{answer} format"""
      match = re.search(r"\\boxed\{([^}]+)\}", action)
      return match.group(1) if match else None

   def step(self, action: str) -> BaseTextEnvStepOutput:
      answer = self._parse_action(action)
      is_correct = answer is not None and answer.strip() == str(self.ground_truth).strip()

      return BaseTextEnvStepOutput(
         observations=[],
         reward=1.0 if is_correct else 0.0,
         done=True,
         metadata={"parsed_answer": answer}
      )
```

## Algorithm

Out of the box, SkyRL supports grpo, gae, rloo, reinforce++

User can implement custom RL algorithm. They have a [DAPO](https://skyrl.readthedocs.io/en/latest/algorithms/dapo.html) example.
