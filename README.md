# Translating Navigation Instructions in Natural Language to a High-Level Plan for Behavioral Robot Navigation

Xiaoxue Zang*, Ashwini Pokle*, Marynel Vázquez, Kevin Chen, Juan Carlos Niebles, Alvaro Soto, Silvio Savarese
* [Project Page]()
* [Paper]()

*\* equal contribution.*
## Setup
Run the following command to install dependencies and libraries, download glove embeddings and dataset for experiments.
``` shell
sh get_started.sh
```
## Usage

### Training

``` shell
python codes/main.py --experiment_name=[unique name for model folder] --data_dir data
```
You can also download the [Trained model](https://stanford.box.com/shared/static/b89tdch98oxjz2iwyxyk1k7x2ntmofnm.zip)
### Test
Evaluate the F1 score and Exact Match on the dev/test dataset, which can be decided by --file_in_path.

``` shell
python codes/main.py --train_dir=experiments/[experiment name] --mode=official_eval --data_dir data --file_in_path [dev/test]
```

--file_out_path is needed if you want to write the prediction out to a file. \[--write_out=True\]

Run the following instruction to evaluate the result and plot the error analysis.
```
python codes/evaluate.py ground_truth_file prediction_file
```

### Show examples
Show the predictions of some examples.
--file_in_path decides the dataset to test with.
There are three valid arguments: dev, test, test_diff_maps, which respectively mean validation set, test-repeated map set, test-new map set.

``` shell
python codes/main.py --train_dir=experiments/[experiment name] --mode=show_examples --data_dir data --file_in_path [dev/test/test_diff_maps] --print_num 10
```

### Examples

Example training commmand:
```
python codes/main.py --experiment_name=my_experiment --mode=train --data_dir my_data
```

Example test command:
```
python codes/main.py --train_dir experiments/my_experiment --mode=official_eval --data_dir my_data
```


# Acknowledgement
*code modified from the Default Final Project for CS224n, Winter 2018*
