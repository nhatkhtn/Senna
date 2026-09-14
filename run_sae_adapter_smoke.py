"""Real-checkpoint verification for the Senna activation adapter.

Run this from the allocated Senna environment.  It reuses the existing
six-camera sample, prompt, preprocessing, and model loader; it does not write
an activation dataset.
"""

import argparse
import json
import os
import sys

import torch

SENNA_ROOT = os.path.dirname(os.path.abspath(__file__))
if SENNA_ROOT not in sys.path:
    sys.path.insert(0, SENNA_ROOT)

from llava.senna.sae_adapter import SennaActivationAdapter  # noqa: E402
from run_sample_senna import (  # noqa: E402
    IMAGE_PROMPT,
    MODEL_PATH,
    SAMPLE_PATH,
    conv_templates,
    find_six_cam_images,
    load_model,
    process_images,
    tokenizer_image_token,
    IMAGE_TOKEN_INDEX,
)
from PIL import Image  # noqa: E402


def _assert_exact(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if not torch.equal(left, right):
        delta = (left.float() - right.float()).abs().max().item()
        raise AssertionError(f"{label} changed (max absolute difference {delta:g})")


def _assert_batched_close(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    """Compare B=2 and B=1 paths, which may select different CUDA kernels."""

    try:
        torch.testing.assert_close(left, right, rtol=5e-2, atol=3e-1)
    except AssertionError as exc:
        delta = (left.float() - right.float()).abs().max().item()
        raise AssertionError(
            f"{label} drifted beyond the B=2 tolerance (max absolute difference {delta:g})"
        ) from exc
    return (left.float() - right.float()).abs().max().item()


def _build_inputs(sample, tokenizer, image_processor, model, device):
    image_files, used_duplicate = find_six_cam_images(sample)
    images = [Image.open(path).convert("RGB") for path in image_files]
    image_sizes = [image.size for image in images]
    images_tensor = process_images(images, image_processor, model.config)
    images_tensor = images_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)

    question = IMAGE_PROMPT + sample["question"]
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    return input_ids, images_tensor, image_sizes, image_files, used_duplicate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--sample", default=SAMPLE_PATH)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Senna SAE verification requires the allocated CUDA GPU")

    with open(args.sample) as handle:
        sample = json.load(handle)
    tokenizer, model, image_processor, device = load_model()
    model.eval()
    input_ids, images, image_sizes, image_files, used_duplicate = _build_inputs(
        sample, tokenizer, image_processor, model, device
    )

    with torch.inference_mode():
        baseline = model(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            use_cache=False,
            return_dict=True,
        )
        adapter = SennaActivationAdapter(
            model,
            args.layer,
            checkpoint=MODEL_PATH,
            tokenizer=getattr(tokenizer, "name_or_path", None),
            prompt_template="llava_v1 + Senna six-view IMAGE_PROMPT",
        )
        captured = adapter.forward(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            use_cache=False,
            return_dict=True,
        )

    _assert_exact(baseline.logits, captured.output.logits, "next-token logits")
    if captured.activations.ndim != 2:
        raise AssertionError("captured activations are not [n, d_model]")
    if captured.activations.shape[-1] != captured.metadata["d_model"]:
        raise AssertionError("activation width does not match metadata d_model")
    final_position = int(captured.metadata["final_prompt_positions"][0])
    expected_first_token = baseline.logits[0, final_position].argmax().item()
    first_ids = model.generate(
        input_ids,
        images=images,
        image_sizes=image_sizes,
        do_sample=False,
        num_beams=1,
        max_new_tokens=1,
        use_cache=True,
    )
    if first_ids.numel() == 0:
        raise AssertionError("native generation returned no token")
    if int(first_ids.reshape(-1)[-1]) != expected_first_token:
        raise AssertionError("final prompt position does not produce the first generated token")

    baseline_ids = model.generate(
        input_ids,
        images=images,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    # A temporary observation-only hook exercises the adapted generation path.
    # Returning None is required so PyTorch keeps the original module output.
    def observe_generation(_module, _inputs, _output):
        pass

    hook = adapter.module.register_forward_hook(observe_generation)
    try:
        adapted_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    finally:
        hook.remove()
    _assert_exact(baseline_ids, adapted_ids, "greedy generated token IDs")

    # Capture the actual vision-tower inputs for clean and changed passes.
    visual_inputs = []
    vision_tower = model.get_model().get_vision_tower()
    visual_hook = vision_tower.register_forward_pre_hook(
        lambda _module, hook_args: visual_inputs.append(hook_args[0].detach().clone())
    )
    try:
        changed_images = images.clone()
        changed_images[:, 0].add_(0.01)
        with torch.inference_mode():
            clean_single = adapter.forward(
                input_ids=input_ids,
                images=images,
                image_sizes=image_sizes,
                use_cache=False,
                return_dict=True,
            )
            changed_single = adapter.forward(
                input_ids=input_ids,
                images=changed_images,
                image_sizes=image_sizes,
                use_cache=False,
                return_dict=True,
            )
    finally:
        visual_hook.remove()
    if len(visual_inputs) < 2 or torch.equal(visual_inputs[-2], visual_inputs[-1]):
        raise AssertionError("changing an image did not reach Senna's visual path")

    # Collection/evaluation is batched.  Use an image-present clean/changed
    # pair and compare it with the two singleton passes.  Hooking may not
    # change the B=2 native operation at all.
    batch_input_ids = input_ids.expand(2, -1)
    batch_images = torch.cat((images, changed_images), dim=0)
    batch_image_sizes = image_sizes * 2
    with torch.inference_mode():
        batch_baseline = model(
            input_ids=batch_input_ids,
            images=batch_images,
            image_sizes=batch_image_sizes,
            use_cache=False,
            return_dict=True,
        )
        batch_captured = adapter.forward(
            input_ids=batch_input_ids,
            images=batch_images,
            image_sizes=batch_image_sizes,
            use_cache=False,
            return_dict=True,
        )
    _assert_exact(
        batch_baseline.logits, batch_captured.output.logits, "batched next-token logits"
    )
    clean_batch_delta = _assert_batched_close(
        batch_captured.output.logits[0],
        clean_single.output.logits[0],
        "clean B=2/B=1 logits",
    )
    changed_batch_delta = _assert_batched_close(
        batch_captured.output.logits[1],
        changed_single.output.logits[0],
        "changed B=2/B=1 logits",
    )
    activation_count = clean_single.activations.shape[0]
    clean_activation_delta = _assert_batched_close(
        batch_captured.activations[:activation_count],
        clean_single.activations,
        "clean B=2/B=1 residuals",
    )
    changed_activation_delta = _assert_batched_close(
        batch_captured.activations[activation_count:],
        changed_single.activations,
        "changed B=2/B=1 residuals",
    )

    duplicate_images = torch.cat((images, images), dim=0)
    duplicate_ids = input_ids.expand(2, -1)
    duplicate_sizes = image_sizes * 2
    batch_ids = model.generate(
        duplicate_ids,
        images=duplicate_images,
        image_sizes=duplicate_sizes,
        do_sample=False,
        temperature=0,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    hook = adapter.module.register_forward_hook(observe_generation)
    try:
        hooked_batch_ids = model.generate(
            duplicate_ids,
            images=duplicate_images,
            image_sizes=duplicate_sizes,
            do_sample=False,
            temperature=0,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    finally:
        hook.remove()
    _assert_exact(batch_ids, hooked_batch_ids, "batched greedy generated token IDs")
    if not (
        torch.equal(batch_ids[0], baseline_ids[0])
        and torch.equal(batch_ids[1], baseline_ids[0])
    ):
        raise AssertionError("batched greedy token IDs disagree with singleton decoding")

    print("Senna SAE adapter verification passed")
    print(f"layer={args.layer} module={captured.metadata['module_path']}")
    print(f"activations={tuple(captured.activations.shape)} final_prompt_position={final_position}")
    print(f"image_count={len(image_files)} duplicated_missing_view={used_duplicate}")
    print(f"greedy_token_ids={baseline_ids.tolist()}")
    print(
        "batch_max_abs_delta="
        f"{max(clean_batch_delta, changed_batch_delta, clean_activation_delta, changed_activation_delta):g}"
    )


if __name__ == "__main__":
    main()
