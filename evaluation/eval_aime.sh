PROMPT_TYPE="qwen-math-cot-standard"
# DATA_NAMES=("math-500" "gsm8k" "minerva_math" "gaokao2023en")
# DATA_NAMES=("math-500" "gsm8k" "minerva_math" "gaokao2023en" "amc23" "olympiadbench" "aime24") 
DATA_NAMES=("aime24")

GPU_IDS=(0)

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Get the directory of the current script for relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


MODEL_DIRS=(
    '/home/yifan50/rl/deepscaler/output/models/aime/aime_model_Qwen2.5-Math-1.5B_dataset_aime_512_epoch_30_bs_512_lr_1e-6_beta_0_entropy_0_mu_1_tau_1e-3_alpha_0.5'
)

# Create separate task files for each GPU
for i in "${!GPU_IDS[@]}"; do
    # Clear any existing task file
    > "${SCRIPT_DIR}/gpu_${GPU_IDS[$i]}_tasks.txt"
done

# Collect all tasks
declare -a ALL_TASKS
task_index=0
for DATA_NAME in "${DATA_NAMES[@]}"; do
    for RAW_MODEL_DIR in "${MODEL_DIRS[@]}"; do

        if [[ "$(basename "$RAW_MODEL_DIR")" != "actor" ]]; then
            MODEL_DIR="${RAW_MODEL_DIR}/actor"
        else
            MODEL_DIR="$RAW_MODEL_DIR"
        fi

        if [[ "$MODEL_DIR" == *"Qwen2.5-Math"* ]]; then
            MAX_NEW_TOKENS=3072
        else
            MAX_NEW_TOKENS=4096
        fi

        N_SAMPLING=1
        
        for MODEL_PATH in $(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -type d); do
            if [[ $(basename "$MODEL_PATH") != "final_checkpoints" ]]; then
                CHECKPOINT_DIR="${MODEL_PATH}"
                OUTPUT_DIR="$(basename "$(dirname "${MODEL_DIR}")")_$(basename "${MODEL_PATH}")"
                
                # Create task line with parameters separated by |
                TASK="${CHECKPOINT_DIR}|${MAX_NEW_TOKENS}|${OUTPUT_DIR}|${DATA_NAME}|${N_SAMPLING}"
                
                # Assign to a GPU in round-robin fashion by adding to its task file
                gpu_index=$((task_index % ${#GPU_IDS[@]}))
                echo "$TASK" >> "${SCRIPT_DIR}/gpu_${GPU_IDS[$gpu_index]}_tasks.txt"
                task_index=$((task_index + 1))
            fi
        done
    done
done

run_on_gpu() {
    local gpu_id=$1
    local task_file="${SCRIPT_DIR}/gpu_${gpu_id}_tasks.txt"
    
    echo "GPU $gpu_id starting with $(wc -l < "$task_file") tasks"
    
    # Process each line in the task file
    while IFS= read -r task_line; do
        # Split by the | delimiter
        IFS='|' read -r MODEL_PATH TOKEN_NUM OUT_DIR DATA N_SAMPLING <<< "$task_line"
        
        echo "GPU $gpu_id running: $MODEL_PATH $TOKEN_NUM $OUT_DIR $DATA"
        echo "Running with: Model=$MODEL_PATH, Tokens=$TOKEN_NUM, Output=$OUT_DIR, Data=$DATA"
        
        if ! CUDA_VISIBLE_DEVICES=$gpu_id bash "${SCRIPT_DIR}/sh/eval.sh" "$PROMPT_TYPE" "$MODEL_PATH" "$TOKEN_NUM" "$OUT_DIR" "$DATA" "$N_SAMPLING"; then
            echo "Task failed on GPU $gpu_id. Terminating all processes."
            # Kill all evaluation processes
            pkill -f "eval.sh"
            # Kill the parent script
            kill -TERM $$
            exit 1
        fi
    done < "$task_file"
    
    echo "GPU $gpu_id finished all tasks"
    # Clean up task file
    rm "$task_file"
}

# Start a process for each GPU with its own task file
for gpu_id in "${GPU_IDS[@]}"; do
    # Only start if there are tasks for this GPU
    if [[ -s "${SCRIPT_DIR}/gpu_${gpu_id}_tasks.txt" ]]; then
        run_on_gpu $gpu_id &
    else
        echo "No tasks for GPU $gpu_id, skipping"
    fi
done

wait
echo "All jobs finished."