METHOD=compressed_sgd
BATCH_SIZE=64
EPOCHS=1
NUM_BITS=1
BINNING=lin
NUM_WORKERS=1
SEED=1917

python3 main.py \
    --weight_decay 1.0 \
    --lr 0.01 0.001 0.001 0.0001 \
    --batchwise_evaluation False \
    --exp_name ${METHOD}_B${BATCH_SIZE}_e${EPOCHS}_${NUM_BITS}bits_${BINNING}Binning_${NUM_WORKERS}workers_Seed${SEED} \
    --model vanilla_cnn \
    --optimizer ${METHOD} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_bits ${NUM_BITS} \
    --binning ${BINNING} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --rand_zero True\
    --dataset MNIST \
    --dataset_root /home/amin/Documents/projects/SignSGDVariant/datasets/mnist
