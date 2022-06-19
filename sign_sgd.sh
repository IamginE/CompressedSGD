METHOD=sign_sgd
BATCH_SIZE=64
EPOCHS=1
SEED=1917

python3 main.py \
    --weight_decay 1.0 \
    --lr 0.01 0.001 0.001 0.0001 \
    --batchwise_evaluation -1 \
    --exp_name ${METHOD}_B${BATCH_SIZE}_e${EPOCHS}_Seed${SEED} \
    --model vanilla_cnn \
    --optimizer ${METHOD} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --rand_zero True \
    --seed ${SEED} \
    --dataset MNIST \
    --dataset_root /home/amin/Documents/projects/SignSGDVariant/datasets/mnist