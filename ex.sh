#/bin/bash

# CIL CONFIG
# NOTE="imagenet_sdp_sigma0_mem_10000_iter_0.125"
NOTE="final_ewc_cifar10_iter2_mem1000"
MODE="ewc"

K_COEFF="4"
TEMPERATURE="0.125"

TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=4
FUTURE_STEPS=8
EVAL_N_WORKER=4
EVAL_BATCH_SIZE=1000
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.5
SEEDS="5"
DATA_DIR=""

GPUS=("0" "1" "2")
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
ONLINE_ITER=2
SIGMA=0
REPEAT=1
INIT_CLS=100
USE_AMP=""

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=1000
    TYPES=("ma" "generated" "web")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=6000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=20000
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="2000 2000 2000 2000 2000"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="2000 2000 2000 2000 2000"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="2000 2000 2000 2000 2000"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="2000 2000 2000 2000 2000"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="2000 2000 2000 2000 2000"
    fi

elif [ "$DATASET" == "PACS" ]; then
    MEM_SIZE=200
    TYPES=("ma" "generated" "web")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=6000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=2000
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="668 614 388"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="648 388 634"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="573 466 631"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="557 712 401"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="570 712 388"
    fi

elif [ "$DATASET" == "OfficeHome" ]; then
    MEM_SIZE=400
    TYPES=("ma" "generated" "web")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=6000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=2000
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="930 833 831 869 894"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="872 757 903 837 988"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="770 980 824 915 868"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="869 870 848 834 936"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="940 892 848 772 905"
    fi

elif [ "$DATASET" == "clear10" ]; then
    MEM_SIZE=4000
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=18000 FEAT_DIM=14 FEAT_MEM_SIZE=96000 #resnet18
    SAMPLES_PER_TASK=3000

elif [ "$DATASET" == "clear100" ]; then
    MEM_SIZE=8000
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=500 
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=60000 FEAT_DIM=14 FEAT_MEM_SIZE=96000 #resnet18
    SAMPLES_PER_TASK=10000

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=2000
    N_SMP_CLS="3" K="3" MIR_CANDS=100
    CANDIDATE_SIZE=200 VAL_SIZE=2
    MODEL_NAME="resnet32" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    # BASEINIT_SAMPLES=60000 FEAT_DIM=4 FEAT_MEM_SIZE=48000
    SAMPLES_PER_TASK=20000

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=10000
    N_SMP_CLS="3" K="3" MIR_CANDS=500
    CANDIDATE_SIZE=1000 VAL_SIZE=2 DATA_DIR="--data_dir /home/vision/smh_new/ILSVRC/Data/CLS-LOC"
    MODEL_NAME="resnet18" EVAL_PERIOD=8000 F_PERIOD=200000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10
    BASEINIT_SAMPLES=769000 FEAT_DIM=14 FEAT_MEM_SIZE=960000 #60% baseinit
    # BASEINIT_SAMPLES=512500 FEAT_DIM=14 FEAT_MEM_SIZE=960000 #40% baseinit
    SAMPLES_PER_TASK=256233 # Number of Tasks: 5

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
