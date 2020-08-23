# python main.py --config 'config/HGD7/HGD7_TEST.yaml' --gpus '0,1' &
# python main.py --config 'config/HGD7/FGSM.yaml' --gpus '2,3' &
# python main.py --config 'config/HGD7/PGD7.yaml' --gpus '4,5' &
# python main.py --config 'config/HGD7/PGD20.yaml' --gpus '6,7'
# python main.py --config 'config/HGD7/MIM20.yaml' --gpus '6,7'

config_files=("HGD7_TEST.yaml" "FGSM.yaml" "PGD7.yaml" "PGD20.yaml" "MIM20.yaml")

for path in "${config_files[@]}"
do
  python main.py --config config/HGD7/$path --gpus 0,1
done



