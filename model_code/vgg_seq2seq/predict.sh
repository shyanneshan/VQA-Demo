cd /data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq;

/data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq/predict.py \
--question "what is the primary abnormality in this image?" \
--imag "/data2/entity/bhy/VQADEMO/VQA-Med-2019/test/synpic53988.jpg" \
--model_dir "weight/dataset.pt" \
--data_dir "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019" \
--traintxt "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/train.txt" \
--trainimg "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/train" \
--valtxt "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/val.txt" \
--valimg "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/val" \
--testtxt "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/test.txt" \
--testimg "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/test" \
