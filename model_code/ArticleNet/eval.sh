cd /data2/entity/bhy/MVQAS/models/ArticleNet;

/data2/entity/bhy/MVQAS/models/MMBERT/venv/bin/python \
-u /data2/entity/bhy/MVQAS/models/ArticleNet/predict.py \
-u /data2/entity/bhy/MVQAS/models/ArticleNet/train.py \
--model_dir 'save/dataset.pt' \
--dataset "dataset" \
--batch-size 32 \
--num-epochs 20