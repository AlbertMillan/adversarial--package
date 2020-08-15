config=('AT_PGD7.yaml', 'AT_PGD')


python main.py --config 'config/AT_PGD7.yaml' --gpus '0,1' &

python main.py --config 'config/AT_PGD20.yaml' --gpus '2,3' &

python main.py --config 'config/AT_KWTA1.yaml' --gpus '4,5' &

python main.py --config 'config/AT_KWTA2.yaml' --gpus '6,7' 