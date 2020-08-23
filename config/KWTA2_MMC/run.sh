# python main.py --config 'config/KWTA2_MMC/KWTA2_TEST.yaml' --gpus '0,1' &
python main.py --config 'config/KWTA2_MMC/FGSM.yaml' --gpus '2,3' &
python main.py --config 'config/KWTA2_MMC/PGD7.yaml' --gpus '4,5' &
python main.py --config 'config/KWTA2_MMC/PGD20.yaml' --gpus '6,7'
python main.py --config 'config/KWTA2_MMC/MIM20.yaml' --gpus '6,7'
