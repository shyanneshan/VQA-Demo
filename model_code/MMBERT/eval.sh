cd /data2/entity/bhy/VQADEMO/model_code/MMBERT;

/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/MMBERT/eval.py \
--data_dir "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019" \
--question "what is the primary abnormality in this image?" \
--imag "/data2/entity/bhy/VQADEMO/uploadimag/7uSwcTru7kb8tos.jpg" \
--model_dir "/data2/entity/bhy/VQADEMO/weights/1028mm/1028mm.pt"