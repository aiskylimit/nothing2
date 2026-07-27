from alpaca_eval.decoders.huggingface_local import huggingface_local_completions
import torch
import datasets
import json
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--model_name", dtype=str)
parser.add_argument("--peft", dtype=str, default=None)

args = parser.parse_args()

model_name = args.model_name
peft_name = args.peft

# Load the 805 AlpacaEval instructions (only needs the dataset, no judge API)
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
instructions = [ex["instruction"] for ex in eval_set]

prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

prompts = [
    prompt_template.format(instruction=instr)
    for instr in instructions
]

completions = huggingface_local_completions(
    prompts=prompts,
    model_name=model_name,
    model_kwargs={"torch_dtype": torch.float16, "device_map": {"": "cuda"}},
    do_sample=True,
    adapters_name=peft_name,
    max_new_tokens=512,
    batch_size=8,
    temperature=1.0,
    top_p=1.0,
)["completions"]

generator_name = peft_name or model_name
outputs = [
    {"instruction": instr, "output": comp, "generator": generator_name}
    for instr, comp in zip(instructions, completions)
]

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{generator_name}.json", "w") as f:
    json.dump(outputs, f, indent=2)