METHOD=sign_sgd
BATCH_SIZE=64
EPOCHS=1
SEED=1917
FULL_EVAL_INTERVAL=50
python3 main.py \
    --weight_decay 1.0 \
    --lr 0.003 0.001 0.0008 0.0005 0.0001 \
    --batchwise_evaluation ${FULL_EVAL_INTERVAL} \
    --exp_name ${METHOD}_B${BATCH_SIZE}_e${EPOCHS}_EvalInterval${FULL_EVAL_INTERVAL}_Seed${SEED} \
    --model vanilla_cnn \
    --optimizer ${METHOD} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --rand_zero True \
    --seed ${SEED} \
    --dataset MNIST \
    --one_plot_optimizer True \
    --dataset_root ./