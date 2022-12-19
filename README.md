# Certifiable Out-of-Distribution Generalization
This is the code repository of the paper *Certifiable Out-of-Distribution Generalization*.
This repository contains the code for getting [OoD-Bench](https://github.com/ynysjtu/ood_bench) results on diversity and correlation shift datasets (Modified from the PyTorch suite [DomainBed](https://github.com/facebookresearch/DomainBed)).

## Environment preparation
#### Recommended PyTorch environment
```
Environment:
    Python: 3.9.12
    PyTorch: 1.10.1+cu111
    Torchvision: 0.11.2+cu111
    CUDA: 11.1
    CUDNN: 8005
    NumPy: 1.21.2
    PIL: 9.1.0
```
The code can theoretically work on arbitrary PyTorch environments, but we do not recommend too old version of PyTorch to avoid the version conflict of the `wilds` package and some errors when installing `torch-scatter`. The experiment results may have a slight fluctuation when changing the environment. This is because of several factors such as the PyTorch version, the CUDA version or the GPU version, and hardware. For reproducing the results exactly, we recommend using our trained weights [here](https://drive.google.com/file/d/1eKp_RNRxCjcO2rLI0TRNl__OBy8eW9A9/view?usp=share_link
).
#### Pre-trained weights preparation
Download ImageNet pre-trained weights of ResNet-18 from https://download.pytorch.org/models/resnet18-5c106cde.pth, and place it under `pretrained_weights`. The directory structure should be:
```
StochasticDisturbanceLearning
├── datasets
├── DomainBed
├── pretrained_weights
├── ...
```

## Data preparation
Most of the datasets (except for CelebA and NICO) can be downloaded by running the script `DomainBed/domainbed/scripts/download.py`, and for NICO we provide a download link for there are some weird extension names in the original dataset.
After the download, place the datasets under `datasets` and make sure the directory structures are as follows:
```
PACS
└── kfold
    ├── art_painting
    ├── cartoon
    ├── photo
    └── sketch
```
```
office_home
├── Art
├── Clipart
├── Product
├── Real World
├── ImageInfo.csv
└── imagelist.txt
```
```
terra_incognita
├── location_38
├── location_43
├── location_46
└── location_100
```
```
WILDS
└── camelyon17_v1.0
    ├── patches
    └── metadata.csv
```
```
MNIST
└── processed
    ├── training.pt
    └── test.pt
```
```
celeba
├── img_align_celeba
└── blond_split
    ├── tr_env1_df.pickle
    ├── tr_env2_df.pickle
    └── te_env_df.pickle
```
```
NICO
├── animal
├── vehicle
└── mixed_split
    ├── env_train1.csv
    ├── env_train2.csv
    ├── env_val.csv
    └── env_test.csv
```
Note: the data split files of CelebA and NICO are already provided under `datasets`.

## Implement the experiments
To simply run the experiments for a certain dataset under a certain algorithm, see `DomainBed/run.py`.  
Example usage:
```python
# Launch
python run.py launch --dataset PACS --algorithm SDL_Gaussian
# If not complete
python run.py delete_incomplete --dataset PACS --algorithm SDL_Gaussian
# List the running status
python run.py list --dataset PACS --algorithm SDL_Gaussian
```
If you want to try for a group of hyper-parameters, firstly edit `DomainBed/sweep/${dataset}/hparams.json` to lock the hyperparameters, then simply run `DomainBed/tuning.py` as the following example:
```python
python tuning.py launch --dataset PACS --algorithm SDL_Gaussian --lr 6e-5 --worst_case_p 0.1
```
For showing the results of running or adjusting, run `DomainBed/collect_run_results.py` or `DomainBed/collect_adjust_results.py` as the following examples:
```python
python collect_run_results.py --dataset PACS --algorithm SDL_Gaussian
python collect_tuning_results.py --dataset PACS --algorithm SDL_Gaussian --lr 6e-5 --worst_case_p 0.1
```
You can check the args in `DomainBed/collect_adjust_results.py` for all the hyper-parameters (the default version is only for SDL algorithms, for others you can write it on your own). The model weights and `results.txt` will be stored in `DomainBed/sweep/${dataset}/outputs/`.
We also provide a python script to list the hyperparameters searching details by connecting `DomainBed/domainbed/scripts/list_top_hparams.py`, the example usage is as follows:
```python
python list.py --dataset PACS --algorithm SDL_Gaussian --test_env 0
```
