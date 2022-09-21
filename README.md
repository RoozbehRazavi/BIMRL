# BIMRL

Code for the paper BIMRL: Brain Inspired Meta Reinforcement Learning 

Seyed Roozbeh Razavi Rohani, Saeed Hedayatian, Mahdiyeh Soleymani 

published at IROS 2022.

```
@inproceedings{roozbeh2020BIMRL,
  title={BIMRL: Brain Inspired Meta Reinforcement Learning},
  author={Seyed Roozbeh Razavi Rohani, Saeed Hedayatian, Mahdiyeh Soleymani },
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  year={2022}}
```

### Requirements

We use PyTorch for this code, and log results using TensorboardX.


### Running an experiment


To run BIMRL on the Mini-Grid experiments use:
```
!python main.py 
```

You can also run other variants of our method due to the flexible implementation. To do so, take a look at config files. 

The results will by default be saved at `./logs`, 
but you can also pass a flag with an alternative directory using `--results_log_dir /path/to/dir`.

The default configs are in the `config/` folder. 
You can overwrite any default hyperparameters using command line arguments.

Results will be written to tensorboard event files, 
and some visualisations will be printed every now and then.
