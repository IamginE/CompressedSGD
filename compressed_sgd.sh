METHOD=compressed_sgd_vote
BATCH_SIZE=64
EPOCHS=3
SEED=1917
FULL_EVAL_INTERVAL=100
NUM_BITS=3
BINNING=lin
NUM_WORKERS=2
python3 main.py \
    --weight_decay 1.0 0.9 0.7 0.5 0.3 \
    --lr 0.003 0.001 0.0005 0.0001\
    --num_bits ${NUM_BITS} \
    --num_workers ${NUM_WORKERS} \
    --binning ${BINNING} \
    --count_usages False \
    --batchwise_evaluation ${FULL_EVAL_INTERVAL} \
    --exp_name ${METHOD}_B${BATCH_SIZE}_n${NUM_BITS}_b${BINNING}_e${EPOCHS}_EvalInterval${FULL_EVAL_INTERVAL}_Seed${SEED}_Workers${NUM_WORKERS} \
    --model vanilla_cnn \
    --optimizer ${METHOD} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --rand_zero True \
    --seed ${SEED} \
    --dataset MNIST \
    --one_plot_optimizer False \
    --dataset_root ./