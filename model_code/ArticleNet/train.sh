cd /data2/entity/bhy/VQADEMO/model_code/ArticleNet;

#/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python txt2json.py --dataset "dataset"&
#/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python createQuery.py --dataset "dataset"&
#/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python process_documents.py --dataset "dataset"&
#/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python preprocess.py --dataset "dataset"&
#/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python dataLoader.py --dataset "dataset"&
/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/ArticleNet/train.py \
--model_dir 'save/dataset.pt' \
--dataset "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019" \
--batch-size 32 \
--num-epoch 20

