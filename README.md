# Attacks & Defenses Pytorch Package
This repo provides an abstract template class to create your own attacks and defenses. It also includes 
implementations of popular adversarial attacks and defenses.
So far, it accounts for first-order attack methods, as well as 
***adversarial training*** [[1]](https://arxiv.org/abs/1706.06083) and
***HGD*** [[2]](https://arxiv.org/abs/1712.02976).
By default, the model employed to generate adversarial examples is WideResNet-28-10 [[3]](https://arxiv.org/abs/1605.07146).
An implementation of this model is retrieved from [[4]](https://github.com/xternalz/WideResNet-pytorch).

## Usage
Download the code 
```
git clone https://github.com/AlbertMillan/adversarial-training-pytorch.git
```

Both attacks and defenses are performed using config files with .yaml extension. 
Modify the variables in the config files to customize your experiment.

### Attacks
To run an adversarial attack such as FGSM, execute the following command:
```
git main.py --config "config/FGSM.yaml" --gpus "0,1"
```

### Defense
To run HGD, execute the following command:
```
git main.py --config "config/AT_HGD.yaml" --gpus "0,1"
```

## ToDo
This repository is still work in progress and thus subject to contain undiscovered bugs. 
Please report encountered issues to the author.
