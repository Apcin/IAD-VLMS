REG = {
 "Qwen2.5-VL-7B": {
   "Base":     {"ds":"outputs/baseline_eval/7b_base_dsmvtec.json",
                "visa":"outputs/baseline_eval/7b_base_visa.json"},
   "SFT":      {"ds":"outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json",
                "visa":"outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json"},
   "SFT+GRPO": {"ds":"outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json",
                "visa":"outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_visa_full_trainprompt.json"},
   "KCR":      {"ds":"outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json",
                "visa":"outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_visa_full_trainprompt.json"},
 },
 "LLaVA-OneVision-7B": {
   "Base":     {"ds":"outputs/llava_ov_7b_zeroshot_eval/eval_dsmvtec_full_yesnouser.json",
                "visa":"outputs/llava_ov_7b_zeroshot_eval/eval_visa_full_yesnouser.json"},
   "SFT":      {"ds":"outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/eval_dsmvtec_full_trainprompt.json",
                "visa":"outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/eval_visa_full_trainprompt.json"},
   "SFT+GRPO": {"ds":"outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_dsmvtec_full_trainprompt_vllm.json",
                "visa":"outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_visa_full_trainprompt_vllm.json"},
   "KCR":      {"ds":"outputs/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/checkpoint-376/eval_dsmvtec_full_trainprompt_vllm.json",
                "visa":"outputs/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/checkpoint-376/eval_visa_full_trainprompt_vllm.json"},
 },
}
STAGES=["Base","SFT","SFT+GRPO","KCR"]
