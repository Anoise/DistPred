export EXP_DIR=./resultsv5
export TASK=uci_bostonv5
export DEVICE_ID=5
export N_SPLITS=20 # 20

export N_STEPS=1
export SERVER_NAME=a4000
export RUN_NAME=run_1
export LOSS=card_conditional
export N_THREADS=4
export CAT_F_PHI=_cat_f_phi
export MODEL_VERSION_DIR=card_conditional_uci_results/${N_STEPS}steps/nn/${RUN_NAME}_${SERVER_NAME}/f_phi_prior${CAT_F_PHI}

python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml --ni

python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test --timesteps 1000