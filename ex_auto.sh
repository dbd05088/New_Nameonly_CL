#!/bin/bash
##SBATCH -p suma_rtx4090
##SBATCH -q big_qos
#SBATCH --gres=gpu:1
##SBATCH -c 64

ulimit -u 200000
source ~/.bashrc
ml purge
conda init bash
conda activate generate

# CIL CONFIG
# NOTE="imagenet_sdp_sigma0_mem_10000_iter_0.125"

# --------------------------IMPORTANT-------------------------- #
MODE="$MODE"
MODEL_NAME="$MODEL_NAME"
DATASET="$DATASET" # cifar10, cifar100, tinyimagenet, imagenet
TYPES=("$TYPE") # Get from environment variable
SEEDS="$SEED" # Get from environment variable
GPUS=("5" "3" "2" "3" "4" "5")
NOTE="iclr_${MODEL_NAME}_${DATASET}_${MODE}"
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
    if [ "$MODEL_NAME" == "vit" ]; then
        LR=1e-4
        echo "Set vit learning rate 1e-4!!!"
    fi
    BASEINIT_SAMPLES=6000 FEAT_DIM=14 FEAT_MEM_SIZE=24000
    SAMPLES_PER_TASK=20000 
    ONLINE_ITER=2
    EVAL_POINT="1000 2000 3000 4000 5000"

elif [ "$DATASET" == "PACS_final" ]; then
    MEM_SIZE=200
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    if [ "$MODEL_NAME" == "vit" ]; then
        LR=1e-4
        echo "Set vit learning rate 1e-4!!!"
    fi
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
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    if [ "$MODEL_NAME" == "vit" ]; then
        LR=1e-4
        echo "Set vit learning rate 1e-4!!!"
    fi
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2
    EVAL_POINT="600 1200 1800 2400"

elif [ "$DATASET" == "DomainNet" ]; then
    MEM_SIZE=8000
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=2000
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    if [ "$MODEL_NAME" == "vit" ]; then
        LR=1e-4
        echo "Set vit learning rate 1e-4!!!"
    fi
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
    MEM_SIZE=8000
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    VAL_PERIOD=500 EVAL_PERIOD=500
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    if [ "$MODEL_NAME" == "vit" ]; then
        LR=1e-4
        echo "Set vit learning rate 1e-4!!!"
    fi
    BASEINIT_SAMPLES=30523 FEAT_DIM=14 FEAT_MEM_SIZE=168000
    SAMPLES_PER_TASK=960
    ONLINE_ITER=3
    EVAL_POINT="9600 19200 28800 38400 48000"

else
    echo "Undefined setting"
    exit 1
fi

LOG_DIR="logs"
mkdir -p $LOG_DIR

LOG_FILE="${LOG_DIR}/iclr_${MODEL_NAME}_${DATASET}_${MODE}_${TYPE}_sd${SEEDS}.log"
echo "SEED: $SEEDS"
python main_new.py --mode $MODE $DATA_DIR \
--dataset $DATASET --unfreeze_rate $UNFREEZE
