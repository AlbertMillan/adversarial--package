
config_files=("AT_PGD7_TEST.yaml" "FGSM.yaml" "PGD7.yaml" "PGD20.yaml" "MIM20.yaml")

for path in "${config_files[@]}"
do
  python main.py --config config/AT/KWTA2_MMC/$path --gpus 6,7
done