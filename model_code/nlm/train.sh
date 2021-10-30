cd /data2/entity/bhy/VQADEMO/model_code/nlm;

/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/nlm/train_vqa.py \
--model-path 'weights/test/test.pkl' \
--data_dir '/data2/entity/bhy/VQADEMO/model_code/nlm/dataset' \
--dataset '/data2/entity/bhy/VQADEMO/model_code/nlm/dataset/train_vqa_dataset.hdf5' \
--val-dataset "/data2/entity/bhy/VQADEMO/model_code/nlm/dataset/val_vqa_dataset.hdf5" \
--test-dataset "/data2/entity/bhy/VQADEMO/model_code/nlm/dataset/test_vqa_dataset.hdf5" \
--batch-size 32 \
--num-epochs 20 \
--vocab-path "/data2/entity/bhy/VQADEMO/model_code/nlm/dataset/vocab_vqa.json" \
--rnn-cell "LSTM" \
--use-w2v \
--embedding-name "PubMed-w2v.txt"
#--use-glove \
#--embedding-name "6B"
