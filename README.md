# Translating Navigation Instructions in Natural Language to a High-Level Plan for Behavioral Robot Navigation

Xiaoxue Zang, Ashwini Pokle, Marynel Vazquez, Kevin Chen, Alvaro Soto, Juan Carlos Niebles, Silvio Savarese
* [Project Page]()
* [Paper]()


**Citing**

If you find this code useful in your work, please cite us:

```
@article{chen2018text2shape,
  title={Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings},
  author={Chen, Kevin and Choy, Christopher B and Savva, Manolis and Chang, Angel X and Funkhouser, Thomas and Savarese, Silvio},
  journal={arXiv preprint arXiv:1803.08495},
  year={2018}
}
```
## Setup
Run the following command to install dependencies and libraries, download glove embeddings and dataset for experiments.
``` shell
sh get_started.sh
```
## Usage

### Training

``` shell
python codes/main.py --experiment_name=[unique name for model folder] --data_dir data --schedule_embed
```

### Test
Evaluate the F1 score and Exact Match on the dev/test dataset, which can be decided by --file_in_path.

``` shell
python codes/main.py --train_dir=experiments/[experiment name] --mode=official_eval --data_dir data --schedule_embed --file_in_path [dev/test]
```

--file_out_path is needed if you want to write the prediction out to a file. \[--write_out=True\]

Run the following instruction to evaluate the result and plot the error analysis.
```
python codes/evaluate.py ground_truth_file prediction_file
```

Example training commmand:
```
python codes/main.py --experiment_name=my_experiment --mode=train --data_dir my_data --schedule_embed
```

Example test command:
```
python codes/main.py --train_dir experiments/my_experiment --mode=official_eval --data_dir my_data --schedule_embed
```


# How to draw the prediction accuracy heatmap?
``` shell
python plot_utils.py [prediction file]
```
Prediction file has [graph triplet number, correct route number, correct or not] in each line.

# Acknowledgement
*code modified from the Default Final Project for CS224n, Winter 2018*
