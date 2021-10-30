cd /data2/entity/bhy/VQADEMO/model_code/ODL;

/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
-u /data2/entity/bhy/VQADEMO/model_code/ODL/predict.py \
--question 'what organ system is displayed in this ct scan?' \
--imag '/data2/entity/bhy/VQADEMO/model_code/ODL/data_RAD/test/synpic16333.jpg' \
--input 'saved_models/san_mevf/model_epoch19.pth' \
--model 'SAN' \
--rnn 'LSTM' \
--dataset 'data_RAD' \
--autoencoder 'True' \
--maml 'True'