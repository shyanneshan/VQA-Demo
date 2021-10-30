cd /data2/entity/bhy/VQADEMO/model_code/nlm;

/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/nlm/evaluate_vqa.py \
--name 'w2v' \
--model-path '/data2/entity/bhy/VQADEMO/weights/w2v' \
--dataset '/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/train_vqa_dataset.hdf5' \
--val-dataset "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/val_vqa_dataset.hdf5" \
--test-dataset "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/test_vqa_dataset.hdf5" \
--vocab-path "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/vocab_vqa.json" \
--test-image "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/test/synpic53988.jpg" \
--test-questions "what is the primary abnormality in this image?" \

