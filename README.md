# BIMRL

Code for the paper "[BIMRL: Brain Inspired Meta Reinforcement Learning](https://arxiv.org/abs/2210.16530)" 

Seyed Roozbeh Razavi Rohani, Saeed Hedayatian, Mahdiyeh Soleymani 

published at IROS 2022.

```
@inproceedings{roozbeh2020BIMRL,
  title={BIMRL: Brain Inspired Meta Reinforcement Learning},
  author={Seyed Roozbeh Razavi Rohani, Saeed Hedayatian, Mahdiyeh Soleymani Baghshah},
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  year={2022}}
```
# Installing Prerequisites

The following packages are required:

- opencv-python==4.5.1
- torch==1.7.1+cu101
- tensorboard==2.4.1
- pynvml==8.0.4
- matplotlib==3.3.2
- tqdm==4.55.1
- scipy==1.6.0
- torchvision==0.8.2+cu101
- gym==0.17.2
- minigrid


# Running an experiment


To run BIMRL on the Mini-Grid experiments use:
```
!python main.py 
```

You can also run other variants of our method due to the flexible implementation. To do so, take a look at config files. 

For instance, for disabling episodic and hebbian memory you can run the commend below:
```
!python main.py --use_memory False
```
Or for disabling only hebbian memory you can run:
```
!python main.py --use_hebb False
```
Also it is possible to only use first or second layer of BRIM module by running:
```
!python main.py --use_rim_level2 False
```
```
!python main.py --use_rim_level3 False
```
There is a lot to explore and maybe you can achieve even better performance, so let's do it - star our repo by the way :)

There are also a number of TODO list, say vision core and lifelong generative module and test sets on MuJoCo benchmark which is not completed yet.

Due to the huge scale of the implementation and since some parts of the code have not been cleaned yet, a handful of files might seems baffling so feel free to contact us through email or the issues part of the repo in case there is a problem ^_^

The results will by default be saved at `./logs`, 
but you can also pass a flag with an alternative directory using `--results_log_dir /path/to/dir`.

The default configs are in the `config/` folder. 
You can overwrite any default hyperparameters using command line arguments as it mentioned eraier.

Results will be written to tensorboard event files, 
and some visualisations will be printed every now and then.

# References

* [VariBAD](https://github.com/lmzintgraf/varibad)
* [Minigrid](https://github.com/Farama-Foundation/Minigrid)
