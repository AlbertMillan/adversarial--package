config_files=("HGD7_MMC_TEST.yaml" "FGSM.yaml" "PGD7.yaml" "PGD20.yaml" "MIM20.yaml")

for path in "${config_files[@]}"
do
  python main.py --config config/HGD7_MMC/$path --gpus 4,5
done