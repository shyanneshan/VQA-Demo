cd /data2/entity/bhy/VQADEMO/model_code/MMBERT;

/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/MMBERT/train.py \
--run_name 'dataset' \
--data_dir 'dataset' \
--save_dir 'dataset' \
--batch_size 16 \
--epochs 10
