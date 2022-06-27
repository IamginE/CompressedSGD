# compressedSGD

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

- Use `plot.py` to plot previous experiments (or parts of which) together. You can, for example, choose a specific Lr and decay combination from one experiments and compare it with a different scenario of the same or another experiment.

- You can also refer to `sgd.sh` and `sign_sgd.sh` sample scripts. However, we recommend modifying `compressed_sgd.sh` with `num_bits 1` to simulate SignSGD. This way you can also try different number of workers (Not supported in sign_sgd's original implementaiton.)

We have the following conventions for plotting:
- The data is stored in the `histories.pkl` files.
- Data is stored as a dictionary with the the keys being the tuple (learning rate, decay) and the metric. 
- For methods that do not us min-max decay the, the default parameter is decay=1.0 (gets ignored during training), but needs to be used to get the data.
- The following metrics exist: 
    - `avg_epoch_loss_hist`: average sample loss per epoch
    - `avg_epoch_acc_hist`: sample accuracy per epoch
    - `batch_loss_hist`: Either batch loss for each batch if `batchwise_evaluation <= 0` or full training data loss if `batchwise_evaluation >= 0`
    - `batch_acc_hist`: Either batch accuracy for each batch if `batchwise_evaluation <= 0` or full training data accuracy if `batchwise_evaluation >= 0`
    - `bin_usage`: Only applicable for compressedSGD and compressedSGDVote, bin usages for discretiztion. Requires `count_usages = True`.

## Structure
- The `logs` directory contains most of our experiments. But the subset of interest is `used_data` directory. The used data folder also contains a file, with all the function calls to generate the plots in used_data/plots. You can simply use them or your own commands at the bottom of `plot.py`.
- On the root directory you can find training scripts as well as the `main.py` which instantiates models, data loaders, optimizers, and trainers and starts training. When finished, it also makes the plots and saves the histories.
- You can find the implementation of optimizers, models, and data loaders in their respective directory. 
- compressedSGDVote simulates distributed compressedSGD
- You can also refer to `used_data/plot_calls.txt` to access the function calls that generate the plots in the paper (from the used_data).

## Commands
- `--seed`: fixes random seed
- `--rand_zero`: only appicable to signSGD, compressedSGD and compressedSGDVote, rounds discretized gradients of value 0 randomly to -1 or 1
- `--exp_name`: name of the folder, where results (plots + histories) get dropped
- `--model`: model used for training
- `--dataset`: dataset used for training
- `--dataset_root`: root directory of the dataset
- `--optimizer`: optimizer to choose from [sgd, sign_sgd, compressed_sgd, compressed_sgd_vote]
- `--batch_size`: batch size used for training
- `--epochs`: number of epochs to train
- `--lr`: space-separated list of learning rates for training
- `--weight_decay`: space-separated list of min-max decays (min_decay, max_decay in the compressedSGD classes), does not get used in signSGD and SGD
- `--binning`: binning strategy to choose from [lin, exp], only applies to compressedSGD and compressedSGDVote
- `--num_bits`: number of bits, used to determine the number of bins, only applies to compressedSGD and compressedSGDVote
- `--num_workers`: number of workers to simulate, only applies to compressedSGDVote
- `--batchwise_evaluation`: After every worker has received this many batches, an evaluation on the entire training set will be done. Set it to `<= 0` to disable evaluation.
- `--count_usages`: counts bin usages and enables to plot them in the end but slows down performance, only applies to compressedSGD and compressedSGDVote
- `--one_plot_optimizer`: determins wether the instant plots for a parameter sweep should be plotted together (True) or seperately (False)