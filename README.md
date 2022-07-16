# E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context  (ECCV 2022)

### Paper(Coming Soon)


[Zizhang Li](https://kyleleey.github.io),
Mengmeng Wang,
Huaijin Pi,
Kechun Xu,
Jianbiao Mei,
Yong Liu

This is the official implementation of the paper "E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context".

## Method overview
<img src=./assets/method.png width="800"  />

## Get started
We run with Python 3.8, you can set up a conda environment with all dependencies like so:

```shell
conda create -n enerv python=3.8
conda activate enerv
pip install -r requirements.txt 
```

## Data
The paper's main experiments were conducted on:
* [data/bunny](./data/bunny) directory contains big buck bunny video sequence
* [UVG](http://ultravideo.fi/#testsequences) video sequences can be found in official website

## Training

### Training experiments
For now, we provide training and evaluation on original NeRV and our E-NeRV, by running following command with different configs and specified ```EXP_NAME```:

```shell
bash scripts/run.sh ./cfgs/E-NeRV-bunny.yaml EXP_NAME 29500
```

The logs and other outputs will be automaically saved to the output directory ```./outputs/EXP_NAME```

## Acknowledgement
Thanks to Hao Chen for his excellent work and implementation [NeRV](https://github.com/haochen-rye/NeRV).

## Contact
If you have any questions, please feel free to email the authors.
