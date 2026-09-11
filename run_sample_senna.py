"""
Smoke test: one forward pass of Senna-VLM on the pinned DriveBench MCQ sample.

Not part of upstream Senna -- a local script added for the ad-uncertainty
project's HPC bring-up. Run from the outer repo root:

    export SLURM_EXPORT_ENV=ALL
    UV_PROJECT_ENVIRONMENT=/tmp/.venv-senna UV_CACHE_DIR=/tmp/.uv_cache \
      srun --jobid=<id> --overlap --export=ALL --ntasks=1 \
        uv run --project Senna python Senna/run_sample_senna.py

Goal is only to prove the model runs to completion and prints a non-empty
string -- Senna is trained for meta-action planning, not free-form MCQ
answers, so the printed answer need not match the ground truth choice.
"""
import glob
import json
import os
import sys

import torch
from PIL import Image

# Make Senna's patched `llava` package importable regardless of cwd.
SENNA_ROOT = os.path.dirname(os.path.abspath(__file__))
if SENNA_ROOT not in sys.path:
    sys.path.insert(0, SENNA_ROOT)

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.senna.senna_llava_llama import SennaLlavaLlamaForCausalLM  # noqa: E402

REPO_ROOT = "/blue/thai/hoangx/projects/ad-uncertainty"
SAMPLE_PATH = os.path.join(REPO_ROOT, "data/test_sample.json")
MODEL_PATH = "rb93dett/Senna"
# The checkpoint's config.json ships an absolute training-machine path here;
# point it at the HF-cached CLIP tower instead.
VISION_TOWER_OVERRIDE = "openai/clip-vit-large-patch14-336"

# Camera order Senna's own data converter uses when building `info['images']`
# (data_tools/senna_nusc_data_converter.py); the 6-view tensor and the 6
# <image> tokens in the text prompt must be in the same order.
CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Same literal prompt template as senna_nusc_data_converter.py:format_qa
# (labels there don't quite line up with CAM_ORDER -- that's an upstream
# quirk we're reproducing as-is, not something we're fixing).
IMAGE_PROMPT = (
    "<FRONT VIEW>:\n<image>\n"
    "<FRONT LEFT VIEW>:\n<image>\n"
    "<FRONT RIGHT VIEW>:\n<image>\n"
    "<BACK LEFT VIEW>:\n<image>\n"
    "<BACK RIGHT VIEW>:\n<image>\n"
    "<BACK VIEW>:\n<image>\n"
)


def find_six_cam_images(sample):
    """Resolve all 6 camera views for the pinned sample's frame.

    The pinned JSON only carries the CAM_BACK path; pull the other 5 views
    from the same nuScenes frame (matched by filename timestamp prefix)
    instead of duplicating CAM_BACK, per the task's stated preference.
    """
    cam_back_rel = sample["image_path"]["CAM_BACK"]
    cam_back_abs = os.path.join(REPO_ROOT, cam_back_rel)
    if not os.path.exists(cam_back_abs):
        raise FileNotFoundError(cam_back_abs)

    fname = os.path.basename(cam_back_abs)
    # n008-2018-08-30-15-52-26-0400__CAM_BACK__1535658934037558.jpg
    scene_prefix, _cam, rest = fname.split("__")
    timestamp = rest.split(".")[0]
    timestamp_prefix = timestamp[:9]  # "1535658934" family per the task note

    samples_dir = os.path.join(REPO_ROOT, "DriveBench/data/nuscenes/samples")

    image_paths = {}
    used_duplicate = False
    for cam in CAM_ORDER:
        pattern = os.path.join(samples_dir, cam, f"{scene_prefix}__{cam}__{timestamp_prefix}*.jpg")
        matches = sorted(glob.glob(pattern))
        if matches:
            image_paths[cam] = matches[0]
        else:
            image_paths[cam] = cam_back_abs
            used_duplicate = True

    return [image_paths[cam] for cam in CAM_ORDER], used_duplicate


def load_model():
    config = AutoConfig.from_pretrained(MODEL_PATH)
    original_tower = getattr(config, "mm_vision_tower", None)
    config.mm_vision_tower = VISION_TOWER_OVERRIDE
    print(f"[info] overriding mm_vision_tower: {original_tower!r} -> {VISION_TOWER_OVERRIDE!r}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    model = SennaLlavaLlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor, device


def main():
    with open(SAMPLE_PATH) as f:
        sample = json.load(f)

    image_files, used_duplicate = find_six_cam_images(sample)
    print("[info] using camera images (in order):")
    for cam, path in zip(CAM_ORDER, image_files):
        print(f"         {cam}: {path}")
    print(f"[info] duplicated CAM_BACK for missing views: {used_duplicate}")

    tokenizer, model, image_processor, device = load_model()

    question = IMAGE_PROMPT + sample["question"]
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = [Image.open(p).convert("RGB") for p in image_files]
    image_sizes = [img.size for img in images]
    images_tensor = process_images(images, image_processor, model.config)
    images_tensor = images_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    print(f"[info] input_ids shape: {tuple(input_ids.shape)}, images shape: {tuple(images_tensor.shape)}")
    print(f"[info] number of <image> tokens in prompt: {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("\n=========== SENNA-VLM SMOKE TEST RESULT ===========")
    print(f"Question: {sample['question']}")
    print(f"Ground truth (MCQ, not required to match): {sample['answer']}")
    print(f"Model output: {outputs!r}")
    print("====================================================\n")

    assert isinstance(outputs, str) and len(outputs) > 0, "model produced an empty answer"
    print("SUCCESS: non-empty answer string produced.")


if __name__ == "__main__":
    main()
