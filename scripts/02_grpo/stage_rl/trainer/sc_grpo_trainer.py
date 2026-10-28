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

import math
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import warnings
from unittest.mock import patch
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl.import_utils import is_vllm_available

from sentence_transformers import SentenceTransformer, models

import copy
from PIL import Image

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def compute_1d_ot(rewards: torch.Tensor) -> torch.Tensor:
    """Map a 1-D tensor of rewards onto N(0,1) via Optimal Transport.

    Each value's rank yields a midpoint quantile p = (rank - 0.5) / N,
    which is mapped through the inverse Gaussian CDF Φ⁻¹(p) =
    sqrt(2)·erfinv(2p-1). Tied rewards receive the AVERAGE of their target
    quantiles — this is what makes the estimator robust to the quantized,
    heavily-tied reward distributions typical of IAD GRPO (group of 4 with
    three of four rollouts sharing a reward bin).

    Ported from OpenVLThinker/EasyR1/verl/trainer/core_algos.py (compute_1d_ot).
    """
    N = rewards.shape[0]
    if N <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, indices = torch.sort(rewards)
    device = rewards.device
    dtype = rewards.dtype

    ranks = torch.arange(1, N + 1, device=device, dtype=torch.float32)
    probabilities = (ranks - 0.5) / N
    target_quantiles = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    target_quantiles = target_quantiles.to(dtype)

    # Tie handling: average target quantiles of identical reward values
    unique_rewards, inverse_indices = torch.unique(sorted_rewards, return_inverse=True)
    if unique_rewards.shape[0] < N:
        target_quantiles_tied = torch.zeros_like(unique_rewards, dtype=dtype)
        target_quantiles_tied.scatter_add_(0, inverse_indices, target_quantiles)
        counts = torch.bincount(inverse_indices).to(dtype)
        target_quantiles_tied = target_quantiles_tied / counts
        target_quantiles = target_quantiles_tied[inverse_indices]

    advantages = torch.zeros_like(rewards)
    advantages[indices] = target_quantiles
    return advantages


class SCGRPOTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        use_vllm_for_gen: bool = True
    ):

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "qwen2-vl" in model_id.lower() or "qwen2_vl" in model_id.lower() or "qwen2vl" in model_id.lower():
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "qwen2.5-vl" in model_id.lower() or "qwen2.5_vl" in model_id.lower() or "qwen2.5vl" in model_id.lower() or "qwen25vl" in model_id.lower():
                filtered_kwargs = {k: v for k, v in model_init_kwargs.items() if k not in ['use_cache']}
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **filtered_kwargs)
            elif "llava-ov" in model_id.lower() or "llava_ov" in model_id.lower() or "llava_si" in model_id.lower():
                filtered_kwargs = {k: v for k, v in model_init_kwargs.items() 
                    if k not in ['use_cache']}
                model = model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **filtered_kwargs)
                 # ===== add padding fix =====
                print("=== Fixing padding for main model ===")
                if hasattr(model, 'processor') and model.processor:
                    if hasattr(model.processor, 'tokenizer'):
                        model.processor.tokenizer.padding_side = 'left'
                        print(f"Set main model processor tokenizer padding_side to: {model.processor.tokenizer.padding_side}")
            elif "llava-next" in model_id.lower() or "llava_next" in model_id.lower() or "llava_1_6" in model_id.lower():
                model_init_kwargs_filtered = {k: v for k, v in model_init_kwargs.items() if k != 'use_cache'} 
                model = LlavaNextForConditionalGeneration.from_pretrained(model, **model_init_kwargs_filtered)
            elif "llava-1_5" in model_id.lower() or "llava_1_5" in model_id.lower():
                model_init_kwargs_filtered = {k: v for k, v in model_init_kwargs.items() if k != 'use_cache'} 
                model = LlavaForConditionalGeneration.from_pretrained(model, **model_init_kwargs_filtered)    
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if os.environ.get("FREEZE_VISION_TOWER","0")=="1":
            import torch as _torch
            n_f=n_t=0
            for _n,_p in model.named_parameters():
                if ".visual." in ("."+_n) and ".merger." not in _n:
                    _p.requires_grad_(False); n_f+=1
                else: n_t+=1
            _vis=getattr(model,"visual",None) or getattr(getattr(model,"model",None),"visual",None)
            _blocks=getattr(_vis,"blocks",None)
            # run the tower under no_grad so no backward recomputation touches its frozen, partitioned blocks
            _orig_fwd=_vis.forward
            def _nograd_forward(*a,**k):
                with _torch.no_grad(): return _orig_fwd(*a,**k)
            _vis.forward=_nograd_forward
            print(f"[freeze] FREEZE_VISION_TOWER=1: {n_f} vision-tower tensors frozen, {n_t} tensors trainable (merger + LM); tower forward wrapped in no_grad; blocks={len(_blocks) if _blocks is not None else '?'}", flush=True)
        self.model_id = model_id
        self.use_vllm = use_vllm_for_gen

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "qwen2-vl" in model_id.lower() or "qwen2_vl" in model_id.lower() or "qwen2vl" in model_id.lower():
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "qwen2.5-vl" in model_id.lower() or "qwen2.5_vl" in model_id.lower() or "qwen2.5vl" in model_id.lower() or "qwen25vl" in model_id.lower():
                filtered_kwargs = {k: v for k, v in model_init_kwargs.items() if k not in ['use_cache']}
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **filtered_kwargs)
            elif "llava-ov" in model_id.lower() or "llava_ov" in model_id.lower() or "llava_si" in model_id.lower():
                filtered_kwargs = {k: v for k, v in model_init_kwargs.items() 
                    if k not in ['use_cache']}
                self.ref_model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **filtered_kwargs)
                # ===== add padding fix =====
                print("=== Fixing padding for reference model ===")
                if hasattr(self.ref_model, 'processor') and self.ref_model.processor:
                    if hasattr(self.ref_model.processor, 'tokenizer'):
                        self.ref_model.processor.tokenizer.padding_side = 'left'
                        print(f"Set ref model processor tokenizer padding_side to: {self.ref_model.processor.tokenizer.padding_side}")
            elif "llava-next" in model_id.lower() or "llava_next" in model_id.lower() or "llava_1_6" in model_id.lower():
                model_init_kwargs_filtered = {k: v for k, v in model_init_kwargs.items() if k != 'use_cache'} 
                self.ref_model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs_filtered)
            elif "llava-1_5" in model_id.lower() or "llava_1_5" in model_id.lower():
                model_init_kwargs_filtered = {k: v for k, v in model_init_kwargs.items() if k != 'use_cache'}
                self.ref_model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs_filtered)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "qwen2-vl" in model_id.lower() or "qwen2_vl" in model_id.lower() or "qwen2vl" in model_id.lower() \
                or "qwen2.5-vl" in model_id.lower() or "qwen2.5_vl" in model_id.lower() or "qwen2.5vl" in model_id.lower() or "qwen25vl" in model_id.lower():
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            elif "llava_ov" in model_id.lower() or "llava-ov" in model_id.lower() or "llava_si" in model_id.lower():
                processing_class = AutoProcessor.from_pretrained(model_id)
    
                # add debugging information after processor initialization
                print("===== LLaVA-OV Image Processor Default Config =====")
                print(f"Image processor type: {type(processing_class.image_processor)}")
                print(f"Image processor config: {processing_class.image_processor}")
                
                # check key parameters
                key_attrs = ['size', 'crop_size', 'image_size', 'do_resize', 'do_center_crop', 'do_pad']
                for attr in key_attrs:
                    if hasattr(processing_class.image_processor, attr):
                        print(f"{attr}: {getattr(processing_class.image_processor, attr)}")
                
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                
                # ===== add padding fix =====
                processing_class.tokenizer.padding_side = 'left'
                print(f"Set LLaVA tokenizer padding_side to: {processing_class.tokenizer.padding_side}")
                # ===== end =====
                processing_class.image_processor.max_pixels = max_pixels
                # processing_class.image_processor.min_pixels = min_pixels
            elif "llava-next" in model_id.lower() or "llava_next" in model_id.lower() or "llava_1_6" in model_id.lower() or "llava_1_5" in model_id.lower() or "llava-1_5" in model_id.lower():
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        
        # vllm or not
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            if self.accelerator.is_main_process:
                if torch.cuda.device_count() == 1:
                    vllm_device = "cuda:0"
                else:
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=0.6,
                        limit_mm_per_prompt={"image": 2},
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=pad_token_id,
            )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, **inputs):
        # print("debug information")
        # if not hasattr(self, '_debug_printed'):
        #     self._debug_printed = True
            
        #     print("=== Token Configuration Debug ===")
            
        #     if hasattr(model, 'config'):
        #         print(f"Model config type: {type(model.config)}")
        #         config_attrs = ['image_token_index', 'im_start_token', 'im_end_token', 
        #                     'vision_start_token', 'vision_end_token', 'img_token_index',
        #                     'image_start_token', 'image_end_token']
                
        #         for attr in config_attrs:
        #             if hasattr(model.config, attr):
        #                 value = getattr(model.config, attr)
        #                 print(f"Model config {attr}: {value}")
            
        #     if hasattr(self, 'processor') and self.processor:
        #         processor = self.processor
        #         print(f"Processor type: {type(processor)}")
                
        #         if hasattr(processor, 'tokenizer'):
        #             tokenizer = processor.tokenizer
        #             print(f"Processor tokenizer type: {type(tokenizer)}")
        #             print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
                    
        #             if hasattr(tokenizer, 'special_tokens_map'):
        #                 print(f"Special tokens map: {tokenizer.special_tokens_map}")
                    
        #             possible_image_tokens = [
        #                 "<image>", "<IMAGE>", "[IMG]", "<img>", "<|image|>",
        #                 "<vision_start>", "<vision_end>", "▁<image>",
        #                 "<im_start>", "<im_end>", "<|vision_start|>", "<|vision_end|>"
        #             ]
                    
        #             vocab = tokenizer.get_vocab()
        #             for token in possible_image_tokens:
        #                 if token in vocab:
        #                     token_id = vocab[token]
        #                     print(f"Found image token '{token}' -> ID: {token_id}")
                    
        #             negative_tokens = {k: v for k, v in vocab.items() if v < 0}
        #             if negative_tokens:
        #                 print(f"Negative token IDs: {negative_tokens}")
                    
        #             large_tokens = {k: v for k, v in vocab.items() if v > 50000}
        #             if large_tokens:
        #                 print(f"Large token IDs (>50000): {dict(list(large_tokens.items())[:10])}")  # top-10
                
        #         processor_attrs = ['image_token_index', 'image_start_token', 'image_end_token',
        #                         'im_start_token', 'im_end_token', 'vision_token_index']
        #         for attr in processor_attrs:
        #             if hasattr(processor, attr):
        #                 value = getattr(processor, attr)
        #                 print(f"Processor {attr}: {value}")
            
        #     trainer_attrs = ['image_token_index', 'processor']
        #     for attr in trainer_attrs:
        #         if hasattr(self, attr):
        #             value = getattr(self, attr)
        #             print(f"Trainer {attr}: {value}")
            
        #     print("================================")
        
        # print("=== Enhanced Debug Info ===")
        
        # if 'input_ids' in inputs:
        #     input_ids = inputs['input_ids']
        #     print(f"Input IDs shape: {input_ids.shape}")
            
        #     negative_tokens = []
        #     large_tokens = []
        #     for i in range(input_ids.shape[0]):
        #         batch_negative = input_ids[i][input_ids[i] < 0]
        #         batch_large = input_ids[i][input_ids[i] > 50000]
        #         if len(batch_negative) > 0:
        #             negative_tokens.extend(batch_negative.tolist())
        #         if len(batch_large) > 0:
        #             large_tokens.extend(batch_large.tolist())
            
        #     unique_negative = list(set(negative_tokens))
        #     unique_large = list(set(large_tokens))
            
        #     if unique_negative:
        #         print(f"Negative token IDs found in input: {unique_negative}")
        #     if unique_large:
        #         print(f"Large token IDs found in input: {unique_large}")
            
        #     print(f"Input IDs range: {input_ids.min().item()} to {input_ids.max().item()}")
            
        #     common_special_tokens = [-200, -100, -1, 32000, 32001, 32002, 151643, 151644, 151645, 128000, 128001, 128256]
        #     for token_id in common_special_tokens:
        #         count = (input_ids == token_id).sum().item()
        #         if count > 0:
        #             print(f"Token ID {token_id} appears {count} times")
            
        #     first_sample = input_ids[0]
        #     print(f"First 20 tokens: {first_sample[:20].tolist()}")
        #     print(f"Last 20 tokens: {first_sample[-20:].tolist()}")
            
        #     for i in range(input_ids.shape[0]):
        #         sample = input_ids[i]
        #         for token_val in torch.unique(sample):
        #             if token_val == 0:
        #                 continue
        #             mask = (sample == token_val)
        #             if mask.sum() > 10:
        #                 print(f"Batch {i}: Token {token_val.item()} appears {mask.sum().item()} times")
        
        # if 'pixel_values' in inputs:
        #     pixel_values = inputs['pixel_values']
        #     print(f"Pixel values shape: {pixel_values.shape}")
        #     if len(pixel_values.shape) == 5:
        #         batch_size, num_patches = pixel_values.shape[:2]
        #         print(f"Batch size: {batch_size}, Num patches per image: {num_patches}")
        
        # print("==========================")
        model_id = self.model_id
        if "llava-ov" in model_id.lower() or "llava" in model_id.lower():
            inputs = self._ensure_left_padding_data(inputs) 
        logits = model(**inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = inputs['input_ids'][:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def _ensure_left_padding_data(self, inputs):
        if 'input_ids' not in inputs:
            return inputs
        
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        
        pad_token_id = 0 
        if hasattr(self, 'processing_class') and hasattr(self.processing_class, 'tokenizer'):
            pad_token_id = self.processing_class.tokenizer.pad_token_id
        
        print(f"=== Checking padding in _get_per_token_logps ===")
        print(f"input_ids shape: {input_ids.shape}")
        print(f"pad_token_id: {pad_token_id}")
        
        batch_size, seq_len = input_ids.shape
        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone() if attention_mask is not None else None
        
        for i in range(batch_size):
            pad_mask = (input_ids[i] == pad_token_id)
            if pad_mask.any():
                pad_positions = pad_mask.nonzero(as_tuple=True)[0]
                first_pad = pad_positions[0].item()
    
                if first_pad < seq_len and torch.all(input_ids[i][first_pad:] == pad_token_id):
                    print(f"Sample {i}: Found right padding starting at position {first_pad}")
                    
                    content = input_ids[i][:first_pad]
                    padding_len = seq_len - first_pad
                    
                    new_input_ids[i] = torch.cat([
                        torch.full((padding_len,), pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
                        content
                    ])
                    
                    if new_attention_mask is not None:
                        new_attention_mask[i] = torch.cat([
                            torch.zeros(padding_len, device=attention_mask.device, dtype=attention_mask.dtype),
                            torch.ones(len(content), device=attention_mask.device, dtype=attention_mask.dtype)
                        ])
                    
                    print(f"Converted sample {i} from right padding to left padding")
                else:
                    print(f"Sample {i}: Already left padding or no padding")
        
        inputs['input_ids'] = new_input_ids
        if new_attention_mask is not None:
            inputs['attention_mask'] = new_attention_mask
        
        print("=== Padding check complete ===")
        return inputs

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        # print("Inputs type and content:")
        # """print"""
        # for i, example in enumerate(inputs):
        #     print(f"Example {i}: {type(example)}")
        #     for key, value in example.items():
        #         print(f"  {key}: {type(value)} - {value}")
        # """print"""
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        if self.use_vllm:
            vllm_prompts_text = copy.deepcopy(prompts_text)
            vllm_prompts = copy.deepcopy(prompts)
        
        import os as _os
        _grpo_max_px = int(_os.environ.get("GRPO_MAX_IMAGE_PIXELS", "0") or "0")
        def _cap_px(im):
            # Pre-resize for processors that ignore image_processor.max_pixels (LLaVA-OV anyres).
            # Mirrors what --max_pixels does for Qwen (area cap, aspect preserved).
            if _grpo_max_px and im.width * im.height > _grpo_max_px:
                f = (_grpo_max_px / (im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.LANCZOS)
            return im
        images = []
        for x in inputs:
            if isinstance(x["image"], list):
                for image in x["image"]:
                    images.append(_cap_px(Image.open(image)) if isinstance(image, str) else _cap_px(image))
            else:
                images = [_cap_px(Image.open(x["image"])) if isinstance(x["image"], str) else _cap_px(x["image"])]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images if len(images) > 0 else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        template_prompt_inputs = prompt_inputs
        prompt_inputs = {
                k: v.repeat(self.num_generations, *[1] * (v.dim() - 1)) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
            }
 
        if self.max_prompt_length is not None:
            prompt_ids = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_mask = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
        else:
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # vllm or not
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if len(images) > 0 and len(vllm_prompts_text) == len(images):
                # test image
                prompts_text_and_vision = [
                    {"prompt": vllm_prompt, "multi_modal_data": {"image": vllm_image}} 
                    for vllm_prompt, vllm_image in zip(vllm_prompts_text, images)
                ]
            elif len(images) > 0 and len(vllm_prompts_text) < len(images):
                num_prompts = len(vllm_prompts_text)
                images_per_prompt = len(images) // len(vllm_prompts_text)
                split_images = [images[i * images_per_prompt: (i + 1) * images_per_prompt] for i in range(num_prompts)]
                # test image and reference image(multi)
                prompts_text_and_vision = [
                    {"prompt": vllm_prompt, "multi_modal_data": {"image": img_list}}
                    for vllm_prompt, img_list in zip(vllm_prompts_text, split_images)
                ]
            else:
                prompts_text_and_vision = [{"prompt": vllm_prompt} for vllm_prompt in vllm_prompts_text]
            
            prompts_text_and_vision = self.num_generations * prompts_text_and_vision
            vllm_prompts = self.num_generations * vllm_prompts

            all_prompts_text_and_vision = gather_object(prompts_text_and_vision)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text_and_vision)

            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(vllm_prompts),
                (self.accelerator.process_index + 1) * len(vllm_prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            prompt_length = prompt_ids.size(1)
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) 
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Non-vLLM path: prompt_inputs is already repeated num_generations times.
            # Generate with num_return_sequences=1 (each repeated prompt gets one completion).
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                temp_gen_config = copy.deepcopy(self.generation_config)
                temp_gen_config.num_return_sequences = 1
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=temp_gen_config,
                )
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        prompt_inputs["input_ids"] = prompt_completion_ids
        prompt_inputs["attention_mask"] = attention_mask

        per_token_logps = self._get_per_token_logps(model, **prompt_inputs)

        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, **prompt_inputs)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, **prompt_inputs)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    # texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, current_step=self.state.global_step , **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Broadcast mean/std for logging regardless of estimator
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        if getattr(self.args, "use_g2rpo", False):
            # G²RPO: per-group rank → quantile → inverse Gaussian CDF, with tie averaging.
            # Optional per-batch whitening pass on top.
            rewards_grouped = rewards.view(-1, self.num_generations)  # [B, G]
            adv_grouped = torch.zeros_like(rewards_grouped)
            for i in range(rewards_grouped.shape[0]):
                adv_grouped[i] = compute_1d_ot(rewards_grouped[i])
            advantages = adv_grouped.reshape(-1)
            if getattr(self.args, "g2rpo_outer_whitening", False) and advantages.numel() > 1:
                advantages = compute_1d_ot(advantages)
        elif getattr(self.args, "use_drgrpo", False):
            # Dr.GRPO (Liu et al. 2503.20783): mean-center only, NO divide-by-group-std.
            # Removes the std-normalisation that, on heavily-tied IAD reward groups, inflates
            # advantages on whatever trivial component still varies (format/location ties) and
            # decouples the gradient from verdict accuracy.
            advantages = rewards - mean_grouped_rewards
        else:
            # Vanilla GRPO: group z-score
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        # --- Flag-gated loss shaping (default-off). With both env flags unset, control falls
        # through to the final else branch, which is the original loss line byte-for-byte, so a
        # crash-restart without the flags behaves identically to the shipped trainer. ---
        _mask_zero_std = os.environ.get("GRPO_MASK_ZERO_STD_GROUPS", "") == "1"
        # GRPO_DRGRPO_TRUE is only meaningful together with --use_drgrpo (REPORT2 §1.3 code-change 1
        # conditions the normalizer on use_drgrpo); without that arg it is deliberately inert.
        _drgrpo_true = os.environ.get("GRPO_DRGRPO_TRUE", "") == "1" and getattr(
            self.args, "use_drgrpo", False
        )
        if _mask_zero_std or _drgrpo_true:
            # Both features act on the single shared loss line, downstream of every estimator
            # branch (g2rpo / drgrpo / z-score), so they apply to all estimators identically.
            loss_mask = completion_mask
            if _mask_zero_std:
                # GRPO_MASK_ZERO_STD_GROUPS=1 (REPORT2 §3.4 rank-1): groups whose within-group
                # reward std == 0 (exact ties) contribute zero loss — zero the completion-mask
                # rows of all G rollouts of each tied group BEFORE the loss mean. Implemented on
                # a local copy (loss_mask) so the metric lines below (completion_length, kl)
                # keep their original semantics; per-row denominators are clamped so fully
                # masked rows contribute exactly 0 rather than 0/0 = NaN.
                # Minimal-faithful reading (recorded per task constraint): the batch mean stays
                # over all B*G rows — masked rows add 0 to the numerator and still count in
                # .mean()'s denominator; no renormalization over surviving rows.
                # std_grouped_rewards was repeat_interleaved above, so each view row is
                # constant and column 0 recovers the per-group std.
                _keep_groups = std_grouped_rewards.view(-1, self.num_generations)[:, 0] != 0
                if bool(_keep_groups.any()):
                    loss_mask = completion_mask * _keep_groups.repeat_interleave(
                        self.num_generations
                    ).unsqueeze(1).to(completion_mask.dtype)
                else:
                    # Denominator guard: every group in the batch is exactly tied, so masking
                    # would zero the whole batch. Fall back to unmasked behavior with a logged
                    # warning (step number in the message defeats the warnings dedup registry).
                    warnings.warn(
                        "GRPO_MASK_ZERO_STD_GROUPS: all groups tied (std==0) at step "
                        f"{self.state.global_step}; falling back to unmasked loss for this batch."
                    )
            if _drgrpo_true:
                # GRPO_DRGRPO_TRUE=1 + --use_drgrpo (REPORT2 §1.1/§1.3, arXiv 2503.20783
                # Listing 1): the second Dr.GRPO deletion — replace the per-response mean
                # 1/|o_i| with the constant generation-budget normalizer 1/MAX_TOKENS
                # (self.max_completion_length, set from args at __init__), then batch mean.
                loss = ((per_token_loss * loss_mask).sum(dim=1) / self.max_completion_length).mean()
            else:
                # Masking only: original per-response mean, with the denominator clamped to >= 1
                # so rows of fully masked (tied) groups contribute exactly 0.
                loss = ((per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        # --- Flag-gated per-group instrumentation (default-off; no-op unless GRPO_GROUP_STATS
        # is set). REPORT2 §3.4 rank-0 / §1.3 code-change 2: closes the batch-mean-only blind
        # spot of the reward metrics above by appending one JSON line per optimization step:
        # {"step", "group_rewards" (per-rollout rewards of each group), "group_stds",
        #  "group_means", "frac_zero_std", "mean_neg_logp"}.
        # mean_neg_logp is the completion-mask-weighted mean of -per_token_logps (detached) —
        # the sampled-token cross-entropy, the entropy proxy REPORT2 rank-0 asks for.
        # Minimal-faithful choices, recorded per task constraint:
        #   * main-process only, and only its LOCAL groups: no new collective op is introduced,
        #     so an env var visible on a subset of ranks can never desynchronize/deadlock them;
        #   * with gradient_accumulation_steps > 1 this appends one line per micro-batch, each
        #     tagged with the same global_step (this method is the only per-batch hook here);
        #   * frac_zero_std uses exact std == 0 (the "exact tie" statistic), Bessel-corrected
        #     std matching the estimator branch above.
        # Must never throw: the whole block is wrapped in try/except.
        _group_stats_path = os.environ.get("GRPO_GROUP_STATS", "")
        if _group_stats_path and self.accelerator.is_main_process:
            try:
                import json as _json

                _grouped_rewards = rewards.view(-1, self.num_generations)
                _group_means = _grouped_rewards.mean(dim=1)
                _group_stds = _grouped_rewards.std(dim=1)
                _mask_sum = completion_mask.sum()
                _mean_neg_logp = (
                    (-per_token_logps.detach() * completion_mask).sum() / _mask_sum.clamp(min=1)
                )
                _stats_line = {
                    "step": int(self.state.global_step),
                    "group_rewards": _grouped_rewards.detach().cpu().tolist(),
                    "group_stds": _group_stds.detach().cpu().tolist(),
                    "group_means": _group_means.detach().cpu().tolist(),
                    "frac_zero_std": float((_group_stds == 0).float().mean().item()),
                    "mean_neg_logp": float(_mean_neg_logp.item()),
                }
                with open(_group_stats_path, "a") as _stats_f:
                    _stats_f.write(_json.dumps(_stats_line) + "\n")
            except Exception as _stats_err:  # instrumentation must never kill a training step
                warnings.warn(f"GRPO_GROUP_STATS logging failed: {_stats_err}")

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()