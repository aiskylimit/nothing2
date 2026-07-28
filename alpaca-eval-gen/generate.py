import torch
import datasets
import json
import os
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--peft", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
args = parser.parse_args()

model_name = args.model_name
peft_name = args.peft

# --- Load model + tokenizer ---
print(f"Loading base model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for correct batched generation on decoder-only models

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda"},
)

if peft_name:
    print(f"Loading LoRA adapter: {peft_name}")
    model = PeftModel.from_pretrained(model, peft_name, torch_dtype=torch.float16)
    model = model.merge_and_unload()

model.eval()

# --- Load AlpacaEval instructions ---
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
instructions = [ex["instruction"] for ex in eval_set]

prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
prompts = [prompt_template.format(instruction=instr) for instr in instructions]

# --- Generate in batches ---
def generate_batch(batch_prompts):
    inputs = tokenizer(
        batch_prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_only = out[:, inputs["input_ids"].shape[1]:]
    return [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in gen_only]

completions = []
batch_size = args.batch_size
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    print(f"Generating {i}/{len(prompts)}...")
    completions.extend(generate_batch(batch))

# --- Save outputs ---
generator_name = peft_name or model_name
safe_name = generator_name.replace("/", "__")

outputs = [
    {"instruction": instr, "output": comp, "generator": generator_name}
    for instr, comp in zip(instructions, completions)
]

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{safe_name}.json", "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Saved {len(outputs)} completions to outputs/{safe_name}.json")