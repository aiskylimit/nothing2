#!/usr/bin/env python3
"""Compare CONTRA-DistiLLM and DistiLLM representation diagnostics."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.hf_data import resolve_data_uri
from distillm.embeddings import SinusoidalPosEmb
from distillm.projector import Projector


DEFAULT_STUDENT = "bachthetrollface/gpt2-120M-contra-distillm-dolly"
DEFAULT_BASELINE = "bachthetrollface/gpt2-120M-distillm-dolly"
DEFAULT_TEACHER = "bachthetrollface/gpt2-1.5B-teacher-dolly"
DEFAULT_VELOCITY = "bachthetrollface/velocity-field-gpt2"


class DenseVelocityField(nn.Module):
    """Architecture used to train bachthetrollface/velocity-field-gpt2."""

    def __init__(self, d_input, d_model, num_distill_layers, n_layers):
        super().__init__()
        self.layer_emb = nn.Embedding(num_distill_layers, d_model)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.input_proj = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1),
                )
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, d_input)

    def forward(self, z_t, t, layer_index):
        is_sequence = z_t.dim() == 3
        if is_sequence:
            batch_size, sequence_length, hidden_size = z_t.shape
            z_t = z_t.reshape(-1, hidden_size)
            t = t[:, None].expand(-1, sequence_length).reshape(-1)
            layer_index = layer_index[:, None].expand(-1, sequence_length).reshape(-1)
        x = self.input_proj(z_t) + self.time_emb(t) + self.layer_emb(layer_index)
        for layer in self.layers:
            x = x + layer(x)
        x = self.output_proj(x)
        if is_sequence:
            x = x.view(batch_size, sequence_length, -1)
        return x


class DollyHiddenStateDataset(Dataset):
    def __init__(self, path, tokenizer, num_examples, max_length, max_prompt_length, token_scope):
        self.examples = []
        with open(resolve_data_uri(str(path)), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.examples.append(json.loads(line))
                if 0 < num_examples <= len(self.examples):
                    break
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.token_scope = token_scope

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        prompt_ids = self.tokenizer(example["prompt"], add_special_tokens=False)[
            "input_ids"
        ][: self.max_prompt_length]
        response_ids = self.tokenizer(
            example.get("output", ""), add_special_tokens=False
        )["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        combined = (prompt_ids + response_ids + [eos_id])[: self.max_length + 1]
        if len(combined) < 2:
            combined = [eos_id, eos_id]
        input_ids = combined[:-1]
        response_start = max(len(prompt_ids) - 1, 0)
        metric_mask = [1] * len(input_ids)
        if self.token_scope == "response":
            metric_mask = [int(i >= response_start) for i in range(len(input_ids))]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "metric_mask": torch.tensor(metric_mask, dtype=torch.bool),
        }


def collate_batch(samples, pad_id):
    max_length = max(sample["input_ids"].numel() for sample in samples)
    input_ids = torch.full((len(samples), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    metric_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = sample["input_ids"].numel()
        input_ids[row, :length] = sample["input_ids"]
        attention_mask[row, :length] = 1
        metric_mask[row, :length] = sample["metric_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "metric_mask": metric_mask & attention_mask.bool(),
    }


def resolve_artifact(source, filename, cache_dir=None):
    source_path = Path(source).expanduser()
    if source_path.is_file():
        return source_path
    if source_path.is_dir():
        artifact = source_path / filename
        if not artifact.exists():
            raise FileNotFoundError(f"Missing {filename} under {source_path}")
        return artifact
    return Path(hf_hub_download(source, filename=filename, cache_dir=cache_dir))


def load_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_flow_modules(source, device, cache_dir=None):
    velocity_path = resolve_artifact(source, "velocity_field.pth", cache_dir)
    projector_path = resolve_artifact(source, "projector.pth", cache_dir)
    velocity_state = load_state_dict(velocity_path)
    projector_state = load_state_dict(projector_path)
    block_indices = {
        int(key.split(".")[1]) for key in velocity_state if key.startswith("layers.")
    }
    velocity = DenseVelocityField(
        d_input=velocity_state["input_proj.weight"].shape[1],
        d_model=velocity_state["input_proj.weight"].shape[0],
        num_distill_layers=velocity_state["layer_emb.weight"].shape[0],
        n_layers=len(block_indices),
    )
    velocity.load_state_dict(velocity_state)
    projector = Projector(
        d_student=projector_state["projector.weight"].shape[1],
        d_teacher=projector_state["projector.weight"].shape[0],
    )
    projector.load_state_dict(projector_state)
    print(f"Loaded velocity field and projector from {source}")
    return velocity.to(device).eval(), projector.to(device).eval()


def get_dtype(name, device):
    if device.type == "cpu" or name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def load_model(path, device, dtype, cache_dir=None):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True, cache_dir=cache_dir
    ).to(device).eval()


def resolve_layer_schedule(student, baseline, teacher, num_flow_layers):
    """Match utils.get_distillation_schedule using the model configs.

    Hidden-state index 0 is the embedding output, while index n is the output
    of transformer block n.  The training config uses 12 student layers, 48
    teacher layers, and six uniformly spaced interior distillation layers.
    """
    student_depth = student.config.num_hidden_layers
    baseline_depth = baseline.config.num_hidden_layers
    teacher_depth = teacher.config.num_hidden_layers
    if baseline_depth != student_depth:
        raise ValueError(
            "Student and baseline must have the same number of layers: "
            f"got {student_depth} and {baseline_depth}"
        )
    student_layers = np.linspace(
        0, student_depth, num_flow_layers + 2, endpoint=True, dtype=int
    )[1:-1].tolist()
    teacher_layers = np.linspace(
        0, teacher_depth, num_flow_layers + 2, endpoint=True, dtype=int
    )[1:-1].tolist()
    return list(zip(student_layers, teacher_layers))


def cosine_sum(x, y):
    return F.cosine_similarity(x.float(), y.float(), dim=-1, eps=1e-8).sum().item()


def linear_cka(x, y, eps=1e-12):
    """Feature-space linear CKA; x and y may have different feature widths."""
    if x.shape[0] < 2:
        return float("nan")
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    hsic = torch.linalg.matrix_norm(x.T @ y).square()
    return (hsic / (torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y) + eps)).item()


def new_stats(times):
    return {
        "tokens": 0,
        "trajectory_distance": 0.0,
        "trajectory_distance_sq": 0.0,
        "target_energy": 0.0,
        "baseline_tokens": 0,
        "baseline_trajectory_distance": 0.0,
        "baseline_trajectory_distance_sq": 0.0,
        "baseline_target_energy": 0.0,
        "velocity": {
            str(t): {"error_sq": 0.0, "target_energy": 0.0, "cosine": 0.0, "tokens": 0}
            for t in times
        },
        "update_flow_cosine": 0.0,
        "update_flow_tokens": 0,
        "cka_student": [],
        "cka_baseline": [],
        "cka_teacher": [],
    }


def sample_for_cka(
    stats, projected_student, projected_baseline, teacher, limit, generator
):
    stored = sum(chunk.shape[0] for chunk in stats["cka_student"])
    remaining = limit - stored
    if remaining <= 0:
        return
    if projected_student.shape[0] > remaining:
        indices = torch.randperm(projected_student.shape[0], generator=generator)[:remaining]
        indices = indices.to(projected_student.device)
        projected_student = projected_student[indices]
        projected_baseline = projected_baseline[indices]
        teacher = teacher[indices]
    stats["cka_student"].append(projected_student.float().cpu())
    stats["cka_baseline"].append(projected_baseline.float().cpu())
    stats["cka_teacher"].append(teacher.float().cpu())


# #### 1. Layer-wise projected student-to-teacher trajectory distance
def measure_trajectory(stats, projected_student, projected_baseline, teacher):
    """Measure student and baseline distances to the same teacher states."""
    displacement = teacher - projected_student
    stats["tokens"] += displacement.shape[0]
    stats["trajectory_distance"] += displacement.norm(dim=-1).sum().item()
    stats["trajectory_distance_sq"] += displacement.square().sum().item()
    stats["target_energy"] += teacher.square().sum().item()
    baseline_displacement = teacher - projected_baseline
    stats["baseline_tokens"] += baseline_displacement.shape[0]
    stats["baseline_trajectory_distance"] += baseline_displacement.norm(dim=-1).sum().item()
    stats["baseline_trajectory_distance_sq"] += baseline_displacement.square().sum().item()
    stats["baseline_target_energy"] += teacher.square().sum().item()
    return displacement


# #### 2. Velocity prediction error for target h_t - h_s
def measure_velocity(stats, velocity, projected_student, teacher, target, layer_id, times):
    count = projected_student.shape[0]
    layer_index = torch.full((count,), layer_id, device=projected_student.device, dtype=torch.long)
    for time_value in times:
        time = torch.full((count,), time_value, device=projected_student.device, dtype=torch.float32)
        h_time = (1.0 - time_value) * projected_student + time_value * teacher
        prediction = velocity(h_time, time, layer_index)
        item = stats["velocity"][str(time_value)]
        item["error_sq"] += (prediction - target).square().sum().item()
        item["target_energy"] += target.square().sum().item()
        item["cosine"] += cosine_sum(prediction, target)
        item["tokens"] += count


# #### 3. Student layer-to-layer update vs. learned-flow cosine
def measure_update_flow_cosine(stats, velocity, projected_current, projected_next, layer_id):
    """Compare P(h_s^{j+1})-P(h_s^j) with v(P(h_s^j), 0, j).

    This is deliberately a depth-wise diagnostic, not an optimizer-step update.
    The velocity field was trained on teacher-directed displacement at each layer,
    so this test asks whether the student's depth trajectory follows that field.
    """
    count = projected_current.shape[0]
    time = torch.zeros(count, device=projected_current.device, dtype=torch.float32)
    layer_index = torch.full((count,), layer_id, device=projected_current.device, dtype=torch.long)
    learned_flow = velocity(projected_current, time, layer_index)
    student_update = projected_next - projected_current
    stats["update_flow_cosine"] += cosine_sum(student_update, learned_flow)
    stats["update_flow_tokens"] += count


# #### 4. Linear CKA after applying f_proj to the student hidden states
def measure_cka(
    stats, projected_student, projected_baseline, teacher, max_samples, generator
):
    sample_for_cka(
        stats,
        projected_student,
        projected_baseline,
        teacher,
        max_samples,
        generator,
    )


@torch.inference_mode()
def evaluate(args):
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = get_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.student_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = DollyHiddenStateDataset(
        args.data_path, tokenizer, args.num_examples, args.max_length,
        args.max_prompt_length, args.token_scope,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda samples: collate_batch(samples, tokenizer.pad_token_id),
    )
    velocity, projector = load_flow_modules(args.velocity_source, device, args.cache_dir)
    student = load_model(args.student_path, device, dtype, args.cache_dir)
    baseline = load_model(args.baseline_path, device, dtype, args.cache_dir)
    teacher = load_model(args.teacher_path, device, dtype, args.cache_dir)
    schedule = resolve_layer_schedule(
        student, baseline, teacher, velocity.layer_emb.num_embeddings
    )
    stats = [new_stats(args.times) for _ in schedule]
    cka_generator = torch.Generator().manual_seed(args.seed)

    for batch_index, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        selected = batch["metric_mask"].to(device)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        student_hidden = student(
            **model_inputs, output_hidden_states=True, use_cache=False
        ).hidden_states
        baseline_hidden = baseline(
            **model_inputs, output_hidden_states=True, use_cache=False
        ).hidden_states
        teacher_hidden = teacher(
            **model_inputs, output_hidden_states=True, use_cache=False
        ).hidden_states
        if selected.any():
            projected = [
                projector(student_hidden[student_layer][selected].float())
                for student_layer, _ in schedule
            ]
            projected_baseline = [
                projector(baseline_hidden[student_layer][selected].float())
                for student_layer, _ in schedule
            ]
            teachers = [
                teacher_hidden[teacher_layer][selected].float()
                for _, teacher_layer in schedule
            ]
            for layer_id, (projected_student, baseline_state, teacher_state) in enumerate(
                zip(projected, projected_baseline, teachers)
            ):
                target = measure_trajectory(
                    stats[layer_id], projected_student, baseline_state, teacher_state
                )
                measure_velocity(
                    stats[layer_id], velocity, projected_student, teacher_state,
                    target, layer_id, args.times,
                )
                if layer_id + 1 < len(projected):
                    measure_update_flow_cosine(
                        stats[layer_id], velocity, projected_student,
                        projected[layer_id + 1], layer_id,
                    )
                measure_cka(
                    stats[layer_id], projected_student, baseline_state, teacher_state,
                    args.cka_max_samples, cka_generator,
                )
        if batch_index % args.log_every == 0 or batch_index == len(dataloader):
            print(f"Processed {batch_index}/{len(dataloader)} batches", flush=True)

    return finalize_results(args, stats, schedule, len(dataset))


def finalize_results(args, stats, schedule, num_examples):
    eps = 1e-12
    layers = []
    for layer_id, ((student_layer, teacher_layer), item) in enumerate(zip(schedule, stats)):
        if not item["cka_student"]:
            raise ValueError(
                "No evaluation tokens were selected. Check --data-path and "
                "--token-scope."
            )
        count = max(item["tokens"], 1)
        baseline_count = max(item["baseline_tokens"], 1)
        projected_student = torch.cat(item["cka_student"], dim=0)
        projected_baseline = torch.cat(item["cka_baseline"], dim=0)
        teacher = torch.cat(item["cka_teacher"], dim=0)
        contra_mean_l2 = item["trajectory_distance"] / count
        baseline_mean_l2 = item["baseline_trajectory_distance"] / baseline_count
        contra_rmse = math.sqrt(item["trajectory_distance_sq"] / count)
        baseline_rmse = math.sqrt(item["baseline_trajectory_distance_sq"] / baseline_count)
        contra_nrmse = math.sqrt(
            item["trajectory_distance_sq"] / (item["target_energy"] + eps)
        )
        baseline_nrmse = math.sqrt(
            item["baseline_trajectory_distance_sq"]
            / (item["baseline_target_energy"] + eps)
        )
        layer = {
            "flow_layer": layer_id,
            "student_layer": student_layer,
            "teacher_layer": teacher_layer,
            "token_count": item["tokens"],
            "trajectory": {
                "student": {
                    "mean_l2": contra_mean_l2,
                    "rmse": contra_rmse,
                    "nrmse": contra_nrmse,
                },
                "baseline": {
                    "mean_l2": baseline_mean_l2,
                    "rmse": baseline_rmse,
                    "nrmse": baseline_nrmse,
                },
                "student_minus_baseline": {
                    "mean_l2": contra_mean_l2 - baseline_mean_l2,
                    "rmse": contra_rmse - baseline_rmse,
                    "nrmse": contra_nrmse - baseline_nrmse,
                },
                "student_relative_improvement": {
                    "mean_l2": (baseline_mean_l2 - contra_mean_l2) / (baseline_mean_l2 + eps),
                    "rmse": (baseline_rmse - contra_rmse) / (baseline_rmse + eps),
                    "nrmse": (baseline_nrmse - contra_nrmse) / (baseline_nrmse + eps),
                },
            },
            "velocity": {},
            "student_update_vs_flow_cosine": (
                item["update_flow_cosine"] / item["update_flow_tokens"]
                if item["update_flow_tokens"] else None
            ),
            "projected_linear_cka": linear_cka(projected_student, teacher),
            "baseline_projected_linear_cka": linear_cka(
                projected_baseline, teacher
            ),
            "cka_sample_count": projected_student.shape[0],
        }
        for time_value in args.times:
            velocity_item = item["velocity"][str(time_value)]
            velocity_count = max(velocity_item["tokens"], 1)
            layer["velocity"][str(time_value)] = {
                "mse": velocity_item["error_sq"] / (velocity_count * teacher.shape[1]),
                "nmse": velocity_item["error_sq"] / (velocity_item["target_energy"] + eps),
                "cosine": velocity_item["cosine"] / velocity_count,
            }
        layers.append(layer)

    def mean_metric(key):
        values = [layer[key] for layer in layers if layer[key] is not None]
        return float(np.mean(values)) if values else None

    aggregate = {
        "trajectory": {
            group: {
                metric: float(np.mean([layer["trajectory"][group][metric] for layer in layers]))
                for metric in ("mean_l2", "rmse", "nrmse")
            }
            for group in (
                "student", "baseline", "student_minus_baseline",
                "student_relative_improvement",
            )
        },
        "student_update_vs_flow_cosine": mean_metric("student_update_vs_flow_cosine"),
        "projected_linear_cka": mean_metric("projected_linear_cka"),
        "baseline_projected_linear_cka": mean_metric(
            "baseline_projected_linear_cka"
        ),
        "velocity": {},
    }
    for time_value in args.times:
        key = str(time_value)
        aggregate["velocity"][key] = {
            metric: float(np.mean([layer["velocity"][key][metric] for layer in layers]))
            for metric in ("mse", "nmse", "cosine")
        }
    return {
        "metadata": {
            "student_label": args.student_label,
            "baseline_label": args.baseline_label,
            "student_path": args.student_path,
            "baseline_path": args.baseline_path,
            "teacher_path": args.teacher_path,
            "velocity_source": args.velocity_source,
            "data_path": str(args.data_path),
            "num_examples": num_examples,
            "token_scope": args.token_scope,
            "times": args.times,
            "layer_pairs": [
                {"flow": i, "student": s, "teacher": t}
                for i, (s, t) in enumerate(schedule)
            ],
        },
        "aggregate": aggregate,
        "layers": layers,
    }


def write_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mechanism_metrics.json"
    csv_path = output_dir / "mechanism_metrics.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    rows = []
    for layer in results["layers"]:
        common = {
            "flow_layer": layer["flow_layer"],
            "student_layer": layer["student_layer"],
            "teacher_layer": layer["teacher_layer"],
        }
        for metric in (
            "student_update_vs_flow_cosine", "projected_linear_cka",
            "baseline_projected_linear_cka",
        ):
            rows.append({**common, "metric": metric, "time": "", "value": layer[metric]})
        for group, trajectory in layer["trajectory"].items():
            for metric, value in trajectory.items():
                rows.append({
                    **common,
                    "metric": f"trajectory_{group}_{metric}",
                    "time": "",
                    "value": value,
                })
        for time_value, velocity_item in layer["velocity"].items():
            for metric, value in velocity_item.items():
                rows.append({
                    **common, "metric": f"velocity_{metric}",
                    "time": time_value, "value": value,
                })
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.csv"
    summary_rows = []
    for label, group, cka_key in (
        (results["metadata"]["student_label"], "student", "projected_linear_cka"),
        (
            results["metadata"]["baseline_label"],
            "baseline",
            "baseline_projected_linear_cka",
        ),
    ):
        trajectory = results["aggregate"]["trajectory"][group]
        summary_rows.append(
            {
                "method": label,
                "trajectory_mean_l2": trajectory["mean_l2"],
                "trajectory_rmse": trajectory["rmse"],
                "trajectory_nrmse": trajectory["nrmse"],
                "projected_linear_cka": results["aggregate"][cka_key],
            }
        )
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(results["aggregate"], indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-path", default=DEFAULT_STUDENT)
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE)
    parser.add_argument("--student-label", default="contra-distillm")
    parser.add_argument("--baseline-label", default="distillm")
    parser.add_argument("--teacher-path", default=DEFAULT_TEACHER)
    parser.add_argument("--velocity-source", default=DEFAULT_VELOCITY)
    parser.add_argument(
        "--data-path",
        default="hf://dvtiendat/contra-data/dolly/valid.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/mechanism")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--token-scope", choices=("response", "all"), default="response")
    parser.add_argument("--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--cka-max-samples", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    write_results(evaluate(args), args.output_dir)


if __name__ == "__main__":
    main()
