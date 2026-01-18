import json
import chz
import sys
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path="data/train-conversations.jsonl"
    )
    # ^^^ Create a dataset from a JSONL file in the same format as
    # tinker_cookbook/example_data/conversations.jsonl
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-examples/sl_basic",
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": 2e-4,
            "lr_schedule": "linear",
            "num_epochs": 1,
            "eval_every": 20,
        }
    )


def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


def build_conversation_dataset(input_file: str, output_file: str):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            question = data['question']
            answer = data['answer']
            conversation = {
                "messages": [
                    {"role": "system", "content": "Solve the following math problem. Provide a step-by-step solution. The final numeric solution is the final line of the solution, preceded by ####. For example: '#### 42'."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            outfile.write(json.dumps(conversation) + '\n')
    print(f"Conversation dataset written to {output_file}")

if __name__ == "__main__":
    build_conversation_dataset("data/train.jsonl", "data/train-conversations.jsonl")
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())