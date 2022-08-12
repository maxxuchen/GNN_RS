#  Maximum weight matching via ML

Code maintainer： Wentao Zhao (wz2543@columbia.edu)

## Instruction
#### Generate sample
`python generate_samples.py`

Optional arguments:
`-s SEED`: random seed used to initialize the pseudo-random number generator
`-j NJOBS`: number of parallel sample-generation jobs.

#### Behavior cloning
`python train.py`

Optional arguments:
`-s SEED`: random seed used to initialize the pseudo-random number generator
`-g GPU`: CUDA GPU id (or -1 for CPU only)
