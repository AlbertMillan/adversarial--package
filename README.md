# Attacks & Defenses Pytorch Package
This repo provides an abstract template class to create your own attacks and defenses. It also includes 
implementations of popular adversarial attacks and defenses.
So far, it accounts for first-order attack methods, as well as 
***adversarial training*** [[1]](https://openreview.net/pdf?id=rJzIBfZAb) and
***HGD*** [[2]](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Liao_Defense_Against_Adversarial_CVPR_2018_paper.pdf).
By default, the model employed to generate adversarial examples is WideResNet-28-10 [[3]](http://www.bmva.org/bmvc/2016/papers/paper087/paper087.pdf).
An implementation of this model is retrieved from [[4]](https://github.com/xternalz/WideResNet-pytorch).

## Usage
Download the code 
```
git clone https://github.com/AlbertMillan/adversarial-training-pytorch.git
cd adversarial--package
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
git test_main.py --config "config/AT_HGD.yaml" --gpus "0,1"
```

## Navigation
Access to the implementation of Wide-ResNet or HGD is provided [here](https://github.com/AlbertMillan/adversarial--package/tree/master/adv_package/loader).

 To check how an HGD model is trained refer to this [file](https://github.com/AlbertMillan/adversarial--package/blob/master/adv_package/defense/evaluation.py)

## ToDo
This repository is still work in progress and thus subject to contain undiscovered bugs. 
Please report encountered issues to the author.

## References
[[1]](https://openreview.net/pdf?id=rJzIBfZAb) A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards  deep  learning  models  resistant  to  adversarial attacks. In: *International Conference on Learning Representations*, 2018

[[2]](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Liao_Defense_Against_Adversarial_CVPR_2018_paper.pdf) F. Liao, M. Liang, Y. Dong, T. Pang, X. Hu, and J. Zhu. Defense  against  adversarial  attacks  using  high-level  repre-sentation guided denoiser. In: *2018 IEEE/CVF Conferenceon Computer Vision and Pattern Recognition*, pages 1778–1787, 2018.

[[3]](http://www.bmva.org/bmvc/2016/papers/paper087/paper087.pdf) S. Zagoruyko and N. Komodakis. Wide Residual Networks. In: Richard C. Wilson, Edwin R. Hancock and William A. P. Smith, editors, *Proceedings of the British Machine Vision Conference (BMVC)*, pages 87.1-87.12. BMVA Press, September 2016.

[[4]](https://github.com/xternalz/WideResNet-pytorch) Wide-ResNet Pytorch implementation (https://github.com/xternalz/WideResNet-pytorch)
