# EDSL: A Biologically Informed and Efficient DNA Sequence Learner
This repository contains the codebase for EDSL, a deep learning framework designed to predict functional genomic events from DNA sequences. The model incorporates biological insights to improve both accuracy and efficiency.

## To train the EDSL model, use the following SLURM job script:
```sbatch jobs/job_train.sbatch.sh```

## Learning rate and other training hyperparameters can be modified in:
```scripts/script_train.py```

## For all model parameters, paths to datasets, and other runtime settings, edit:
```configs/config_train.yml```

## Evaluation scripts are provided in files with names ending in _eval.
Run these scripts to assess the model's performance on test data.
