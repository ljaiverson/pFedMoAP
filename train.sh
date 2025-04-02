#!/bin/bash

# MODEL=fedavg
# TRAINER=PromptFL

# MODEL=FedOTP
# TRAINER=GLP_OT

MODEL=pFedMoAP
TRAINER=PFEDMOAP

# MODEL=fedavg
# TRAINER=CLIP


CFG=rn50  # config file

# experiment settings
DATASET=food101
USERS=10
FRAC=1 # paticipation rate
LR=0.001
ROUND=10
USEALL=False # use all for non-fewshot experiments
SHOTS=16
SEED=0

# pFedMoAP settings
SPARSE_SELECTION=nearest
LMBDA=0.5
SCALING=10.0 # higher for more different attention weights
NUM_EXPERTS=10


# CoOp settings (fixed for FL experiments)
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
if [ "$TRAINER" = "CLIP" ]; then
    CFG_FILE=./configs/trainers/PromptFL/${CFG}.yaml
    CTXINIT=True
else
    CFG_FILE=./configs/trainers/${TRAINER}/${CFG}.yaml
    CTXINIT=False
fi

DIR=output/${DATASET}/${MODEL}/${CFG}/nctx${NCTX}/${USERS}users_seed${SEED} # insert more hyperparameter into file name

python federated_main.py \
    --model ${MODEL} \
    --trainer ${TRAINER} \
    --num_users ${USERS} \
    --frac ${FRAC} \
    --lr ${LR} \
    --round ${ROUND} \
    --num_shots ${SHOTS} \
    --train_batch_size ${SHOTS} \
    --config-file ${CFG_FILE} \
    --output-dir ${DIR} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --ctx_init ${CTXINIT} \
    --lmbda ${LMBDA} \
    --num_experts ${NUM_EXPERTS} \
    --sparse_selection ${SPARSE_SELECTION} \
    --scaling ${SCALING} \
    --seed ${SEED}

# done

