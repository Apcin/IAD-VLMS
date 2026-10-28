# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Type
from functools import partial

from datasets import load_dataset

from configs import GRPOConfig
from trainer import SCGRPOTrainer
from reward import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config


logger = logging.getLogger(__name__)

# ZeRO-3 guard: SigLIP's fan-based _init_weights crashes on zero3-partitioned
# (0/1-D placeholder) tensors during from_pretrained. All weights are loaded from
# the checkpoint afterwards, so init values are irrelevant — no-op the fan-based
# init for such tensors. Affects only LLaVA-OV (SigLIP tower); Qwen path untouched.
try:
    from transformers.models.siglip import modeling_siglip as _msig
    _orig_variance_scaling = _msig.variance_scaling_
    def _vs_zero3_guard(tensor, *a, **k):
        if tensor.dim() < 2 or tensor.numel() == 0:
            return tensor
        return _orig_variance_scaling(tensor, *a, **k)
    _msig.variance_scaling_ = _vs_zero3_guard
except Exception:
    pass

@dataclass
class GRPOScriptArguments(ScriptArguments):

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "reasoning"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning'"},
    )
    trainer_cls: Type = field(
        default=SCGRPOTrainer, 
        metadata={"help": "Trainer class"}
    )
    use_vllm_for_gen: str = field(
        default="true", 
        metadata={"help": "Whether to use vllm for fast generation"}
    )
    use_system_prompt: str = field(
        default="false", 
        metadata={"help": "Whether to use system_prompt (True) or use question_template instead (False)"}
    )
    image_path: Optional[str] = field(
        default="/data", 
        metadata={"help": "Path to images"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    single_img: int = field(
        default=1,
        metadata={"help": "Whether to use single image mode"}
    )
    prompt_style: str = field(
        default="grpo",
        metadata={"help": "User-prompt style: 'grpo' (default, the IAD-R1 'Are there any defects in the query image?' question template) or 'sft' (the product-specific SFT/eval prompt 'Analyze the provided image of the {product}...'). 'grpo' preserves all existing behaviour."}
    )

def main(script_args, training_args, model_args):

    use_system_prompt = False if script_args.use_system_prompt == "false" else True
    use_vllm_for_gen = False if script_args.use_vllm_for_gen == "false" else True

    if script_args.single_img == 1: # 0-shot training set
        GENERAL_SYSTEM_PROMPT = (
            'You are an expert in detecting anomalies in image. Your task is to detect if there are any anomalies in the test image.'

            'If you find anomalies in the test image, structure your response with the following format:'
            '<think>[Your process of observation and reasoning is here]</think>'
            '<location>[The location of the anomaly in the image]</location>'
            '<type>[The type of anomaly in the image]</type><answer>[Your final answer is here(yes or no)]</answer>'
 
            'If no anomalies are detected in the test image, structure your response with the following format:'
            '<think>[Your process of observation and reasoning is here]</think>'
            '<answer>[Your final answer is here(yes or no)]</answer>'
            '{Question}'      
        )
        SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT

        GENERAL_QUESTION_PROMPT = (
            'You are an expert in detecting defects in image. Your task is to detect if there are any defects in the test image.'
            '{Question}'
        )
    elif script_args.single_img == 0: # 1-shot training set
        GENERAL_SYSTEM_PROMPT = (
        'You are an expert in detecting anomalies in images. I will provide you with two images: a reference image (first) showing a normal object without defects, and a test image (second) that needs inspection.'

        'Your task is to compare these images and determine if there are any anomalies in the test image. Use the reference image as a baseline for what is considered normal.'

        'If you find anomalies in the test image, structure your response with the following format:'
            '<think>[Your process of observation and reasoning is here]</think>'
            '<location>[The location of the anomaly in the image]</location>'
            '<type>[The type of anomaly in the image]</type><answer>[Your final answer is here(yes or no)]</answer>'
 
            'If no anomalies are detected in the test image, structure your response with the following format:'
            '<think>[Your process of observation and reasoning is here]</think>'
            '<answer>[Your final answer is here(yes or no)]</answer>'

        'Remember that the first image is always the reference (normal) image, and the second image is the test image that needs inspection.'
        '{Question}'    
        )
        SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT

        GENERAL_QUESTION_PROMPT = (
                'You are an expert in detecting defects in image. I will provide you with two images: a reference image (first) showing a normal object without defects, and a test image (second) that needs inspection.'
                'Your task is to compare these images and determine if there are any anomalies in the test image. Use the reference image as a baseline for what is considered normal.'
                '{Question}'
            )
    else:
        raise ValueError("The single_img parameter can only be 0 or 1")


    

    QUESTION_PROMPT = GENERAL_QUESTION_PROMPT


    reward_funcs_registry = {
        "accuracy": accuracy_reward,
        "format": consistency_reward,
        "reasoning": reasoning_reward,
        "reasoning_gemini": reasoning_reward_gemini,
        "reasoning_gemini_rank": reasoning_reward_gemini_rank,
    }

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json'):
        dataset = load_dataset('json', data_files=script_args.dataset_name)
        def make_conversation(example, image_path=None, use_system_prompt=False, prompt_style="grpo"):
            SPEC_QUESTION_PROMPT = QUESTION_PROMPT

            # Support our dataset format (image_path/question/answer)
            # and the original IAD-R1 format (image/problem/solution)
            if "image_path" in example and example["image_path"]:
                # Our format: image_path is already an absolute path
                images = [example["image_path"]]
            elif "image" in example and example["image"]:
                if isinstance(example["image"], list):
                    images = []
                    for item in example["image"]:
                        if isinstance(item, str):
                            images.append(os.path.join(image_path, item))
                        elif isinstance(item, dict):
                            images.append(os.path.join(image_path, item["path"]))
                        else:
                            raise TypeError("Unsupported Format.")
                elif isinstance(example["image"], str):
                    images = [os.path.join(image_path, example["image"])]
                elif isinstance(example["image"], dict):
                    images.append(os.path.join(image_path, example["image"]["path"]))
                else:
                    raise TypeError("Unsupported Format.")
            else:
                return {}

            # Use "question" (our format) or "problem" (IAD-R1 format)
            problem = example.get("question") or example.get("problem", "")

            # solution: use full GT response with tags so reward functions can extract
            # <type>, <location>, <answer> from it. Falls back to gt_label if answer absent.
            if example.get("answer"):
                solution = example["answer"]
            else:
                gt = example.get("gt_label", example.get("solution", ""))
                solution = gt if re.search(r"<answer>", gt) else f"<answer>{gt}</answer>"

            # SFT/eval-aligned prompt (opt-in): the product-specific "Analyze the
            # provided image of the {product}..." prompt used by SFT and eval, as a
            # single user turn with no system message. Default 'grpo' path below is
            # untouched, so all other runs keep the IAD-R1 question template.
            if prompt_style in ("sft", "sft_sys"):
                product = example.get("product")
                if not product:
                    parts = images[0].replace("\\", "/").split("/")
                    product = next((parts[i+1] for i, p in enumerate(parts)
                                    if p == "images" and i+1 < len(parts)), "object")
                sft_text = (
                    f"Analyze the provided image of the {product}. "
                    "Determine if there are any anomalies present. "
                    "If an anomaly is detected, specify its type and location, "
                    "and provide a detailed reasoning for your conclusion."
                )
                if prompt_style == "sft_sys":
                    # eval-aligned: renders identically to the default eval prompt
                    # (system "Please answer by yes or no" + user "\n"+sft_text).
                    # NOTE: system content is a LIST (of one text part), not a bare
                    # string, so it matches the user content type — otherwise
                    # datasets.map -> pyarrow fails with "cannot mix list and
                    # non-list". The chat template renders it the same either way.
                    return {
                        "prompt": [
                            {"role": "system", "content": [{"type": "text", "text": "Please answer by yes or no"}]},
                            {
                                "role": "user",
                                "content": [
                                    *[{"type": "image"} for _ in images],
                                    {"type": "text", "text": f"\n{sft_text}"},
                                ],
                            },
                        ],
                        "image": images,
                        "solution": solution,
                    }
                # "sft": user-only, no system message (previous-run reproduction)
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image"} for _ in images],
                                {"type": "text", "text": sft_text},
                            ],
                        },
                    ],
                    "image": images,
                    "solution": solution,
                }

            if use_system_prompt:
                return {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image"} for _ in images],
                                {"type": "text", "text": problem},
                            ],
                        },
                    ],
                    "image": images,
                    "solution": solution,
                }
            else:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image"} for _ in images],
                                {"type": "text", "text": SPEC_QUESTION_PROMPT.format(Question=problem)},
                            ],
                        },
                    ],
                    "image": images,
                    "solution": solution,
                }

        dataset = dataset.map(partial(make_conversation, image_path=script_args.image_path, use_system_prompt=use_system_prompt, prompt_style=script_args.prompt_style))
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    trainer = script_args.trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        use_vllm_for_gen=use_vllm_for_gen
    )

    # Train model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    # Save and push to hub
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
