#!/usr/bin/env python3
"""Evaluate one model's generated images against the published test manifest.

The evaluator uses all 2,015 test pairs and joins predictions by sample_id
instead of assuming a model-specific filename convention.
"""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


def safe_join(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative path below {root}: {relative}")
    root = root.resolve()
    joined = (root / path).resolve()
    if not joined.is_relative_to(root):
        raise ValueError(f"Path leaves {root}: {relative}")
    return joined


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def resolve_inputs(metadata, predictions, image_root, predictions_root):
    rows = read_rows(metadata)
    predicted = read_rows(predictions)
    if not rows or not predicted:
        raise ValueError("Metadata and predictions must both contain rows")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in metadata")
    mapping = {}
    for row in predicted:
        key = row["sample_id"]
        if key in mapping:
            raise ValueError(f"Duplicate prediction for sample_id {key}")
        mapping[key] = row["generated_path"]
    if set(ids) != set(mapping):
        missing = sorted(set(ids) - set(mapping))
        extra = sorted(set(mapping) - set(ids))
        raise ValueError(f"Predictions must cover exactly the metadata IDs; missing={missing[:5]}, extra={extra[:5]}")
    pairs = []
    for row in rows:
        paths = (safe_join(image_root, row["before"]), safe_join(image_root, row["after"]),
                 safe_join(predictions_root, mapping[row["sample_id"]]))
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        pairs.append(paths)
    return pairs


def image_tensor(path, device, size=512):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def clip_features(paths, device, batch_size, scorer):
    import torch

    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            images = [image_tensor(path, torch.device("cpu")) for path in batch]
            processed = scorer.processor(images=images, return_tensors="pt", padding=True)
            vectors = scorer.model.get_image_features(processed["pixel_values"].to(device))
            features.append((vectors / vectors.norm(p=2, dim=-1, keepdim=True)).cpu())
    return torch.cat(features)


def clip_scores(features, prompt, device, scorer):
    import torch

    with torch.no_grad():
        processed = scorer.processor(text=[prompt], return_tensors="pt", padding=True)
        text_feature = scorer.model.get_text_features(
            processed["input_ids"].to(device), processed["attention_mask"].to(device)
        )
        text_feature = (text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)).cpu()
        return 100 * (features @ text_feature.T).flatten()


def legacy_batch_mean(scores, batch_size=16):
    """Mean of batch means used by the original PITI prompt scripts."""
    return sum(float(scores[start:start + batch_size].mean())
               for start in range(0, len(scores), batch_size)) / ((len(scores) + batch_size - 1) // batch_size)


def perceptual_metrics(pairs, device, model):
    import lpips
    import numpy as np
    import torch
    from torch_fidelity import calculate_metrics

    scorer = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_size = 256 if model == "piti" else 512
    distances = []
    with torch.no_grad():
        for _, target, generated in pairs:
            generated_tensor = image_tensor(generated, device, lpips_size).float().unsqueeze(0) / 127.5 - 1
            target_tensor = image_tensor(target, device, lpips_size).float().unsqueeze(0) / 127.5 - 1
            distances.append(float(scorer(generated_tensor, target_tensor)))
    del scorer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # FID/KID use the original files. Names are unique
    # because a source image can appear under more than one pair ID.
    with tempfile.TemporaryDirectory() as temporary:
        gen_dir, target_dir = Path(temporary) / "generated", Path(temporary) / "target"
        gen_dir.mkdir()
        target_dir.mkdir()
        for index, (_, target, generated) in enumerate(pairs):
            shutil.copyfile(generated, gen_dir / f"{index:06d}{generated.suffix.lower()}")
            shutil.copyfile(target, target_dir / f"{index:06d}{target.suffix.lower()}")
        distribution = calculate_metrics(input1=str(gen_dir), input2=str(target_dir),
                                         cuda=device.type == "cuda", fid=True, kid=True,
                                         kid_subset_size=min(len(pairs), 1000),
                                         save_cpu_ram=True, verbose=False)
    return {"lpips_mean": float(np.mean(distances)),
            "lpips_std": float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0,
            "fid": float(distribution["frechet_inception_distance"]),
            "kid_mean": float(distribution["kernel_inception_distance_mean"]),
            "kid_std": float(distribution["kernel_inception_distance_std"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--metadata", type=Path, default=root / "metadata/test_pairs.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Key in config/prompts.json")
    parser.add_argument("--prompts", type=Path, default=root / "config/prompts.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Metric device; defaults to CPU to avoid unexpected GPU use")
    parser.add_argument("--check-only", action="store_true", help="Validate joins and files without loading metric packages")
    parser.add_argument("--semantic-only", action="store_true", help="Skip LPIPS, FID, and KID")
    parser.add_argument("--perceptual-only", action="store_true", help="Skip CLIP and delta-CLIP")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.semantic_only and args.perceptual_only:
        parser.error("Choose at most one of --semantic-only and --perceptual-only")
    config = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.model not in config["model_specific"]:
        parser.error(f"Unknown model: {args.model}")
    pairs = resolve_inputs(args.metadata, args.predictions, args.image_root, args.predictions_root)
    if args.check_only:
        print(f"Validated {len(pairs)} complete pairs for {args.model}")
        return
    if len(pairs) < 2:
        parser.error("FID/KID require at least two pairs")

    import torch
    from torchmetrics.multimodal import CLIPScore
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    result = {"model": args.model, "pairs": len(pairs), "device": str(device),
              "metrics": {} if args.semantic_only else perceptual_metrics(pairs, device, args.model), "prompts": {}}
    if args.perceptual_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Evaluated {len(pairs)} pairs for {args.model}; wrote {args.output}")
        return
    generated = [paths[2] for paths in pairs]
    before = [paths[0] for paths in pairs]
    no_prior = args.model in config["no_prior_image"]
    clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device).eval()
    generated_features = clip_features(generated, device, args.batch_size, clip_scorer)
    before_features = None if no_prior else clip_features(before, device, args.batch_size, clip_scorer)
    groups = {"model_specific": config["model_specific"][args.model], **config["planning"]}
    piti_legacy = {}
    for name, prompts in groups.items():
        if isinstance(prompts, str):
            prompts = [prompts]
        scores = []
        for prompt in prompts:
            generated_scores = clip_scores(generated_features, prompt, device, clip_scorer)
            generated_score = max(0.0, float(generated_scores.mean()))
            before_score = None if no_prior else max(0.0, float(
                clip_scores(before_features, prompt, device, clip_scorer).mean()))
            scores.append({"prompt": prompt, "clip": generated_score,
                           "delta_clip": None if no_prior else generated_score - before_score})
            if args.model == "piti":
                piti_legacy.setdefault(name, []).append(legacy_batch_mean(generated_scores))
        result["prompts"][name] = {
            "clip": sum(item["clip"] for item in scores) / len(scores),
            "delta_clip": None if no_prior else sum(item["delta_clip"] for item in scores) / len(scores),
            "individual": scores,
        }
    if args.model == "piti":
        # The published PITI script did not clear its batch-score list between
        # prompts. Keep the per-prompt score above and expose the historical
        # aggregate separately for comparison with the published tables.
        model_specific = piti_legacy["model_specific"]
        result["prompts"]["model_specific"]["published_legacy_clip"] = sum(
            sum(model_specific[:index]) / index
            for index in range(1, len(model_specific) + 1)
        ) / len(model_specific)
        planning = [piti_legacy[name][0] for name in config["planning"]]
        for index, name in enumerate(config["planning"], start=1):
            result["prompts"][name]["published_legacy_clip"] = sum(planning[:index]) / index
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(pairs)} pairs for {args.model}; wrote {args.output}")


if __name__ == "__main__":
    main()
