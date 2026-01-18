"""
requires TINKER_API_KEY environment variable to be set.
"""

import tinker
from tinker.types import SamplingParams
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook import model_info
import argparse
import json

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to the input GSM8K file. Except a JSONL file with 'question' field.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file. The output will be a JSONL file with 'question' and 'answer' fields.")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name to use for inference.")
    parser.add_argument("--model-path", type=str, help="Path to the model checkpoint to use for inference. If not provided, the base model will be used.")
    return parser.parse_args()


def main():
    args = parse_args()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.base_model, model_path=args.model_path)

    tokenizer = tokenizer_utils.get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    print(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()
    print(f"Stop sequences: {stop_sequences}")
    sampling_params = SamplingParams(max_tokens=512, temperature=0.5, stop=stop_sequences)

    # First loop: read input and process
    questions = []
    futures = []
    with open(args.input, 'r') as infile:
        for line in tqdm(infile):
            data = json.loads(line)
            question = data['question']
            questions.append(question)
            messages = [
                {"role": "system", "content": "Solve the following math problem. Provide a step-by-step solution. The final numeric solution is the final line of the solution, preceded by ####. For example: '#### 42'."},
                {"role": "user", "content": question}
            ]
            prompt = renderer.build_generation_prompt(messages)
            output_future = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1)
            futures.append(output_future)

    # Second loop: write to output
    with open(args.output, 'w') as outfile:
        assert len(futures) == len(questions)
        for future, question in tqdm(zip(futures, questions), total=len(futures)):
            output = future.result()
            sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
            print(f"Sampled message: {sampled_message}")
            print(f"Parse success: {parse_success}")
            if parse_success:
                answer = sampled_message['content']
            else:
                answer = "[parsing failed]"
            result = {"question": question, "answer": answer}
            outfile.write(json.dumps(result) + '\n')


if __name__== "__main__":
    main()

