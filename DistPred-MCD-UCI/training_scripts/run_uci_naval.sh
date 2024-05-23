export EXP_DIR=./results_naval
export DEVICE_ID=3

export N_STEPS=1000
export SERVER_NAME=a4000
export RUN_NAME=run_2
export LOSS=card_conditional
export TASK=uci_naval
export N_SPLITS=20
export N_THREADS=4

export CAT_F_PHI=_cat_f_phi
export MODEL_VERSION_DIR=uci_results/${N_STEPS}steps/nn/${RUN_NAME}_${SERVER_NAME}/f_phi_prior${CAT_F_PHI}

python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml --ni >$EXP_DIR/${MODEL_VERSION_DIR}/logs/train.log

# python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test --timesteps 1000