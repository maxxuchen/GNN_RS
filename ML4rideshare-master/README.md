# ML4rideshare
Code maintainer： Wentao Zhao (wz2543@columbia.edu)

## Installation
System requirement: Ubuntu 20.04, 21.04
```bash
git clone https://github.com/LoganZhao1997/ML4rideshare
cd ML4rideshare
source env.sh
```

## Run the code
### Behavior cloning
It is actually supervised learning rather than behavior cloning. I used the wrong name when I started coding. 
#### Generate samples for bc
```bash
python bc/bc_generate_samples.py
```
Optional arguments: 
`-s SEED`: random seed used to initialize the pseudo-random number generator;
`-j NJOBS`: number of parallel sample-generation jobs.

#### Do the training
```bash
python bc/bc_train.py PROBLEM
```
`PROBLEM`: choose `rr_graph` or `vr_graph` to train
where `rr_graph` denotes the first-phase and `rr_graph` denotes the second phase. 
Optional arguments:
`-s SEED`: random seed used to initialize the pseudo-random number generator;
`-g GPU`: CUDA GPU id (or -1 for CPU only). 
After training, the network parameters will be saved in `bc/trained_models/best_params.pkl`.


#### Evolution strategy
#### Generate samples for es
Please use the jupyter notebook `Generate_ES_Sample.ipynb` in the folder `es` to generate samples.
Note that please make sure the number of orders is small. Otherwise, it may take a long time to evaluate.

#### Do the training
Please note that bc and es use different samples for training.
```bash
python es/es_main.py PROBLEM
```
`PROBLEM`: choose `rr_graph` or `vr_graph` to train
Please make sure that there is parameters saved in path `bc/trained_models/best_params.pkl`.
