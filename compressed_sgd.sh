METHOD=compressed_sgd
BATCH_SIZE=64
EPOCHS=1
SEED=1917
FULL_EVAL_INTERVAL=-1
NUM_BITS=2
BINNING=lin
python3 main.py \
    --weight_decay 1.0 0.8 0.7 0.5 0.3 \
    --lr 0.003 0.001 0.0005 0.0001 \
    --num_bits ${NUM_BITS} \
    --binning ${BINNING} \
    --count_usages True \
    --batchwise_evaluation ${FULL_EVAL_INTERVAL} \
    --exp_name ${METHOD}_B${BATCH_SIZE}_n${NUM_BITS}_l${BINNING}$_e${EPOCHS}_EvalInterval${FULL_EVAL_INTERVAL}_Seed${SEED} \
    --model vanilla_cnn \
    --optimizer ${METHOD} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --rand_zero True \
    --seed ${SEED} \
    --dataset MNIST \
    --one_plot_optimizer False \
    --dataset_root ./