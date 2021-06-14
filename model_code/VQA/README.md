# VQA
Visual Question Answering

The proposed visual question answering system is based on the variational auto-encoders architecture and designed so that it can take a radiology image and a question as input and generate an answer as output.



![VQGR model](https://github.com/sarrouti/VQA/blob/master/vqa_ve-1.jpg)

## Requirements
gensim==3.0.0\
nltk==3.4.5\
numpy==1.12.1\
Pillow==6.2.0\
progressbar2==3.34.3\
h5py==2.8.0\
torch==0.4.0\
torchvision==0.2.0\
torchtext==0.2.3\
jupyter==1.0.0

install Python requirements:
```
pip install -r requirements.txt
```
## Downloads and Setup
Once you clone this repo, run the vocab.py, store_dataset.py, train.py and evaluate.py file to process the dataset, to train and evaluate the model.
```shell
$ python vocab.py
$ python store_dataset.py
$ python train_vqa.py
$ python evaluate_vqa.py
```
## Citation
If you are using this repository or a part of it, please cite our paper:
```
@inproceedings{sarrouti2020nlm,
  title={NLM at VQA-Med 2020: Visual question answering and generation in the medical domain},
  author={Sarrouti, Mourad},
  year={2020},
  organization={CLEF}
}
```

## Contact
For more information, please contact me on sarrouti.mourad[at]gmail.com.

