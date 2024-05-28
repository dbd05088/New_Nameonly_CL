# #!/bin/bash
# #SBATCH --time=20:00:00
# #SBATCH --gres=gpu:1
# #SBATCH -c 64

# source ~/.bashrc
# ml purge
# conda init bash
# conda activate generate

# CIL CONFIG
# NOTE="imagenet_sdp_sigma0_mem_10000_iter_0.125"

# --------------------------IMPORTANT-------------------------- #
MODE="er"
MODEL_NAME="resnet18"
DATASET="cct" # cifar10, cifar100, tinyimagenet, imagenet
NOTE="iclr_${MODEL_NAME}_${DATASET}_${MODE}"
TYPES=("ma" "static_cot_50_sdxl", "generated_equalweight")
SEEDS="5"
GPUS=("4" "1" "2" "3" "4")
# --------------------------IMPORTANT-------------------------- #
echo "MODE: $MODE"
echo "MODEL_NAME: $MODEL_NAME"
echo "DATASET: $DATASET"
echo "NOTE: $NOTE"

TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=4
FUTURE_STEPS=8
EVAL_N_WORKER=4
EVAL_BATCH_SIZE=500
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.5
DATA_DIR=""
K_COEFF="4"
TEMPERATURE="0.125"

SIGMA=0
REPEAT=1
INIT_CLS=100
USE_AMP=""

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=1000
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=500
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=6000 FEAT_DIM=14 FEAT_MEM_SIZE=24000
    SAMPLES_PER_TASK=20000 
    ONLINE_ITER=2
    EVAL_POINT="1000 2000 3000 4000 5000"

elif [ "$DATASET" == "PACS_final" ]; then
    MEM_SIZE=200
    # TYPES=("sdbp") # web_RMD_temp_0_5_W web_inverse_temp_0_5_WF "web_RMD_temp_3" "inverseprob" "bottomk" "topk") #"sampling_4" "sampling_2" "sampling_0_5" "sampling_0_25" "sampling_0_125" #"equalweighted" "ensembled_RMD_0_5_modelwise" "ensembled_RMD_1_modelwise" "ensembled_RMD_3_modelwise" "ensembled_RMD_classwise_temp_0_5" "ensembled_RMD_classwise_temp_1" "PACS_final_ensembled_RMD_classwise_temp_3") #("ensembled_samplewise_RMD_0_5" "ensembled_samplewise_RMD_1" "ensembled_samplewise_RMD_3" "ensembled_samplewise_RMD_5") #("ensembled_RMD_temp1" "ensembled_RMD_temp2" "ensembled_RMD_temp5" "ensembled_RMD_temp10")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2

    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="458 764 1333"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="638 948 1333"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="520 1013 1333"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="641 950 1333"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="455 764 1333"
    fi

elif [ "$DATASET" == "cct" ]; then
    MEM_SIZE=400
    # TYPES=("web_RMD_w_normalize_clip_90_temp_0_5") #"train_ma" "sdxl_diversified" "web" "generated" "sdxl_diversified_nofiltering" "RMD_classwise_temp_0_5" "RMD_classwise_temp_1" "RMD_classwise_temp_3"
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2
    EVAL_POINT="600 1200 1800 2400"

elif [ "$DATASET" == "DomainNet" ]; then
    MEM_SIZE=8000
    # TYPES=("sdbp") # "newsample_equalweight"
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=4000
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=30523 FEAT_DIM=14 FEAT_MEM_SIZE=168000
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=3
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="10037 20073 30338 40393 50273"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="10189 20304 30413 40177 50273"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="9825 19943 30186 40109 50273"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="10276 20398 30423 40359 50273"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="9918 20137 30290 40184 50273"
    fi

elif [ "$DATASET" == "NICO" ]; then
    MEM_SIZE=800 #1500
    # TYPES=("sdbp") # "newsample_equalweight"
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=500
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=30523 FEAT_DIM=14 FEAT_MEM_SIZE=168000
    SAMPLES_PER_TASK=960
    ONLINE_ITER=2
    EVAL_POINT="960 1920 2880 3840 4800"

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    for index in "${!TYPES[@]}"
    do
        CUDA_VISIBLE_DEVICES=${GPUS[$index]} nohup python main_new.py --mode $MODE $DATA_DIR \
        --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE \
        --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --samples_per_task $SAMPLES_PER_TASK \
        --rnd_seed $RND_SEED --val_memory_size $VAL_SIZE --type_name "${TYPES[$index]}" \
        --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        --lr $LR --batchsize $BATCHSIZE --mir_cands $MIR_CANDS --eval_point "${EVAL_POINT}" \
        --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER \
        --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE \
        --baseinit_samples $BASEINIT_SAMPLES --spatial_feat_dim $FEAT_DIM --feat_memsize $FEAT_MEM_SIZE &
    done
done
