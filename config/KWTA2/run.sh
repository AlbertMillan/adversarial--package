python main.py --config 'config/KWTA2/KWTA2_TEST.yaml' --gpus '0,1' &
python main.py --config 'config/KWTA2/FGSM.yaml' --gpus '2,3' &
python main.py --config 'config/KWTA2/PGD7.yaml' --gpus '4,5' &
python main.py --config 'config/KWTA2/PGD20.yaml' --gpus '6,7'
