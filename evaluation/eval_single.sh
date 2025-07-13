PROMPT_TYPE="qwen-math-cot-standard"
# DATA_NAMES=("olympiadbench")
export CUDA_VISIBLE_DEVICES="8"
MAX_NEW_TOKENS=3072

# MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_NAME="olympiadbench"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B"
OUTPUT_DIR=$(basename "${MODEL_NAME_OR_PATH}")  
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_NEW_TOKENS $OUTPUT_DIR $DATA_NAME

# MODEL_DIRS=(
#     # "/home/yifan50/rl/deepscaler/output/models/yifan_model_Qwen2.5-Math-1.5B_dataset_math_level3to5_8486_ds_average_rewards_debug_epoch_30_bs_512_lr_1e-6_length_3072_G_8_beta_1e-3_system_prompt_std_ckp_True/actor"
#     )

# for DATA_NAME in "${DATA_NAMES[@]}"; do
#     for MODEL_DIR in "${MODEL_DIRS[@]}"; do
#         for MODEL_PATH in $(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -type d); do
#             if [[ $(basename "$MODEL_PATH") != "final_checkpoints" ]]; then
#                 echo "Processing checkpoint directory: $MODEL_PATH"
#                 if [ -d "$MODEL_PATH" ]; then 
#                     CHECKPOINT_DIR="${MODEL_PATH}"
#                     OUTPUT_DIR="$(basename "$(dirname "${MODEL_DIR}")")_$(basename "${MODEL_PATH}")"
#                     echo "Evaluating model at: $CHECKPOINT_DIR"
#                     bash sh/eval.sh $PROMPT_TYPE $CHECKPOINT_DIR $MAX_NEW_TOKENS $OUTPUT_DIR $DATA_NAME
#                 fi
#             else
#                 echo "Skipping final_checkpoints directory: $MODEL_PATH"
#             fi
#         done
#     done
# done
