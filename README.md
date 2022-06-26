# SignSGDVariant

## Setup
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Use Cases
- You can refer to `compressed_sgd.sh` for a sample training script. The results will be stored at `./logs` directory. You can find five plots and a pickle file that can later be used to compare different experiments on different metrics with each other.
    ```bash
    source env/bin/activate
    source compressed_sgd.sh
    ```
    You can use 
    ```Python
    python main.py --help
    ```
    to learn more about the arguments.

- Use `plot.py` to plot previous expereriments (or parts of which) together. You can, for example, choose a specific Lr and decay combination from one experiments and compare it with a different scenario of the same or another experiment.

- You can also refer to `sgd.sh` and `sign_sgd.sh` sample scripts. However, we recommend modifying `compressed_sgd.sh` with `num_bits 1` to simulate SignSGD. This way you can also try different number of workers (Not supported in sign_sgd's original implementaiton.)

We have the following conventions for plotting:
- The data is stored in the `histories.pkl` files.
- Data is stored as a dictionary with the the keys being the tuple (learning rate, decay) and the metric. 
- For methods that do not us min-max decay the, the default parameter is decay=1.0 (gets ignored during training), but needs to be used to get the data.
- The following metrics exist: 
- - `avg_epoch_loss_hist`: average sample loss per epoch
- - `avg_epoch_acc_hist`: sample accuracy per epoch
- - `batch_loss_hist`: Either batch loss for each batch, if `batchwise_evaluation <= 0` or full training data loss if `batchwise_evaluation >= 0`
- - `batch_acc_hist`: Either batch accuracy for each batch, if `batchwise_evaluation <= 0` or full training data accuracy if `batchwise_evaluation >= 0`
- - `bin_usage`: Only applicable for compressedSGD and compressedSGDVote, bin usages for discretiztion. Requires `count_usages = True`

'avg_epoch_loss_hist': avg_epoch_loss_hist, 'avg_epoch_acc_hist': avg_epoch_acc_hist, 
                'batch_loss_hist': batch_loss_hist, 'batch_acc_hist': batch_acc_hist, 
                'bin_usage':
## Structure
- The `logs` directory contains most of our experiments. But the subset of interest is `used_data` directory. The used data folder also contains a file, with all the function calls to generate the plots in used_data/plots. You can simply use them or your own commands at the bottom of `plot.py`.
- On the root directory you can find training scripts as well as the `main.py` which instantiates models, data loaders, optimizers, and trainers and starts training. When finished, it also makes the plots and saves the histories.
- You can find the implementation of optimizers, models, and data loaders in their respective directory. 
- You can also refer to `used_data/plot_calls.txt` to access the function calls that generate the plots in the paper (from the used_data).

## Commands
