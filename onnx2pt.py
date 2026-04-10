import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from backbones import get_model


COMMON_PREFIXES = ("module.", "model.", "backbone.")


def strip_common_prefixes(key: str) -> str:
    changed = True
    out = key
    while changed:
        changed = False
        for prefix in COMMON_PREFIXES:
            if out.startswith(prefix):
                out = out[len(prefix):]
                changed = True
    return out


def load_onnx_initializers(onnx_path: str) -> Dict[str, torch.Tensor]:
    model = onnx.load(onnx_path)
    out: Dict[str, torch.Tensor] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        # Ensure a writable contiguous array before conversion to tensor.
        arr = np.array(arr)
        tensor = torch.from_numpy(arr)
        out[init.name] = tensor
    return out


def build_normalized_index(tensors: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for name in tensors.keys():
        idx[strip_common_prefixes(name)].append(name)
    return idx


def pick_tensor_for_key(
    model_key: str,
    model_shape: torch.Size,
    initializers: Dict[str, torch.Tensor],
    normalized_index: Dict[str, List[str]],
) -> Tuple[torch.Tensor, str]:
    # 1) Exact key match.
    if model_key in initializers and initializers[model_key].shape == model_shape:
        return initializers[model_key], model_key

    # 2) Common prefixed key variants.
    for prefix in COMMON_PREFIXES:
        cand = prefix + model_key
        if cand in initializers and initializers[cand].shape == model_shape:
            return initializers[cand], cand

    # 3) Match by stripped key when it is unambiguous by shape.
    normalized = strip_common_prefixes(model_key)
    candidates = normalized_index.get(normalized, [])
    shape_matched = [
        c for c in candidates if initializers[c].shape == model_shape]
    if len(shape_matched) == 1:
        selected = shape_matched[0]
        return initializers[selected], selected

    return None, ""


def convert_onnx_to_pt(
    onnx_path: str,
    output_path: str,
    network: str,
    num_features: int = 512,
    strict: bool = False,
) -> None:
    print(f"Loading ONNX model: {onnx_path}")
    initializers = load_onnx_initializers(onnx_path)
    print(f"Found {len(initializers)} initializers in ONNX graph")

    print(
        f"Building target backbone: network={network}, num_features={num_features}")
    model = get_model(network, dropout=0.0, fp16=False,
                      num_features=num_features)
    model_state = model.state_dict()

    normalized_index = build_normalized_index(initializers)

    converted_state = {}
    used_initializer_names = set()
    matched_count = 0
    unmatched_keys = []

    for key, value in model_state.items():
        tensor, picked_name = pick_tensor_for_key(
            model_key=key,
            model_shape=value.shape,
            initializers=initializers,
            normalized_index=normalized_index,
        )

        if tensor is None:
            # Keep default values for missing items (e.g. num_batches_tracked).
            converted_state[key] = value
            unmatched_keys.append(key)
            continue

        converted_state[key] = tensor.to(dtype=value.dtype)
        used_initializer_names.add(picked_name)
        matched_count += 1

    unused_initializers = sorted(
        set(initializers.keys()) - used_initializer_names)

    print(f"Matched model params/buffers: {matched_count}/{len(model_state)}")
    print(f"Unmatched model keys: {len(unmatched_keys)}")
    if unmatched_keys:
        print("First unmatched keys:")
        for name in unmatched_keys[:20]:
            print(f"  - {name} (shape={tuple(model_state[name].shape)})")

    print(f"Unused ONNX initializers: {len(unused_initializers)}")
    if unused_initializers:
        print("First unused initializers:")
        for name in unused_initializers[:20]:
            print(f"  - {name} (shape={tuple(initializers[name].shape)})")

    if strict and unmatched_keys:
        raise RuntimeError(
            "Strict mode enabled and some model keys could not be matched from ONNX. "
            "Use non-strict mode or verify --network/--num-features are correct."
        )

    torch.save(converted_state, output_path)
    print(f"Saved converted state_dict to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ONNX initializers into a PyTorch state_dict .pt for LVFace backbones"
    )
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to input .onnx model")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output .pt file")
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help=(
            "Backbone name for get_model(), e.g. vit_b_dp005_mask_005, vit_s, r50"
        ),
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=512,
        help="Embedding dimension used by backbone head (default: 512)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any target state_dict key cannot be matched from ONNX",
    )

    args = parser.parse_args()
    convert_onnx_to_pt(
        onnx_path=args.onnx,
        output_path=args.output,
        network=args.network,
        num_features=args.num_features,
        strict=args.strict,
    )
