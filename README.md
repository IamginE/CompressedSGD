# SignSGDVariant

## Setup
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Use Cases
1. You can refer to `compressed_sgd.sh` for a sample training script. The results will be stored at `./logs` directory. You can find five plots and a pickle file that can later be used to compare different experiments on different metrics with each other.
```bash
source env/bin/activate
source compressed_sgd.sh
```
You can use 
```Python
python main.py --help
```
to learn more about the arguments.

2. Use `plot.py` to plot previous expereriments (or parts of which) together. You can, for example, choose a specific Lr and decay combination from one experiments and compare it with a different scenario of the same or another experiment.

3. You can also refer to `sgd.sh` and `sign_sgd.sh` sample scripts. However, we recommend modifying `compressed_sgd.sh` with `num_bits 1` or `weight_decay 0` to simulate SignSGD. This way you can also try different number of workers (Not supported in sign_sgd's original implementaiton.)
