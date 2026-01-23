# Model Training Using Fireworks

## SFT

SFT is no-code. User can only control the dataset and hyper-params.

[docs](https://docs.fireworks.ai/fine-tuning/fine-tuning-models)

## RL

RL is low-code. User writes an Evaluator. The rest is handled by platform.

Evaluator = Training dataset + grading function. [Example](https://github.com/eval-protocol/quickstart-gsm8k/blob/main/evaluation.py)

[docs](https://docs.fireworks.ai/fine-tuning/quickstart-math)

User has two options on where the Evaluator runs:
- either on the user's local machine, or
- user can provide a Dockerfile, and the evaluator runs inside docker containers, hosted by fireworks.

### Advanced RL Features

#### Remote Rollout

With remote rollout, the user's infrastructure handles rollout and grading, and fireworks handles training.

When is it useful?
- Multi-turn conversations. The default low-code RL only supports single-turn.
- The agent uses private tools, like access to private databases

[docs](https://docs.fireworks.ai/fine-tuning/connect-environments)

#### Secure Training

In usual SFT, users upload datasets to Fireworks.

With secure training, user put the data in the user's Google Cloud Service (GCS) buckets.

Fireworks download the data from GCS when fine-tuning, and deletes the data when fine-tuning completes.

Useless because it depends on GCS, not Azure.