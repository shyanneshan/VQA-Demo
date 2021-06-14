from __future__ import print_function
import os

import argparse

from sklearn.metrics import precision_score, recall_score, f1_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from pickle import dump
from keras.utils import plot_model
from data_utils import Utility
import pandas as pd
import numpy as np
import os
import nltk


parser = argparse.ArgumentParser()

# Session parameters.
#str
parser.add_argument('--pretrained', default="/home/wxl/Documents/VQADEMO/model_code/VQA-Med/vgg16_weights_tf_dim_ordering_tf_kernels.h5", type=str, help="pretrained vgg model weights")
parser.add_argument('--glove', default="/home/wxl/Documents/VQADEMO/model_code/VQA-Med/glove/glove.6B.300d.txt", type=str, help="glove file")
parser.add_argument('--mode', default="train", type=str, help="train or predict")
parser.add_argument('--run_name', default="vggvqa", type=str, help="run name for wandb")
parser.add_argument('--question', type=str,
                    default='what modality is shown?',
                    help='test txt')
parser.add_argument('--imag', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/VQAMed2019Test/VQAMed2019_Test_Images/synpic54082.jpg',
                    help='test imag')
parser.add_argument('--train_text_file', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/train/train.txt',
                    help='train txt(imag|question|answer)')
parser.add_argument('--valid_text_file', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/valid/valid.txt',
                    help='valid txt(imag|question|answer)')
parser.add_argument('--test_text_file', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/test/test.txt',
                    help='test txt(imag|question|answer)')
parser.add_argument('--train_image_file', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/train/train/',
                    help='train imag')
parser.add_argument('--valid_image_file', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/valid/valid/',
                    help='valid imag')
parser.add_argument('--test_image_file', type=str,
                    default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/test/test/',
                    help='test imag')
parser.add_argument('--train_pct', type=float, required=False, default=1.0, help="fraction of train samples to select")
parser.add_argument('--valid_pct', type=float, required=False, default=1.0,
                    help="fraction of validation samples to select")
parser.add_argument('--test_pct', type=float, required=False, default=1.0, help="fraction of test samples to select")
parser.add_argument('--model_path', type=str, default='weights/tf1/',
                        help='Path for saving trained models')
parser.add_argument('--save_dir', type=str,
                    default='',
                    help='save model dictionary')
#int
# emb_size = 300
# hidden_nodes = 1024
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--hidden_nodes', type=int, default=1024)
parser.add_argument('--num_epochs', type=int, default=20)#1000
parser.add_argument('--batch_size', type=int, default=256)#20

#float
parser.add_argument('--lambda-gen', type=float, default=1.0,
                        help='coefficient to be added in front of the generation loss.')

args = parser.parse_args()

import sys
# sys.path.append('/home/wxl/Documents/VQADEMO/model_code/VQA-Med')

batch_size = args.batch_size # Batch size for training.
'''
原先是1000
'''
epochs = args.num_epochs#100 initial Number of epochs to train for.
def extract_data(file):
    ids,imag, ques, answ = [],[],[],[]
    f = open(file,'r')
    lines = f.readlines()
    count=1
    for line in lines:
        ids.append(count)
        imag.append(line.split('|')[0])
        ques.append(line.split('|')[-2])
        answ.append(line.strip().split('|')[-1])
        count+=1
    f.close()
    temp={0:ids,
          1:imag,
          2:ques,
          3:answ}
    df=pd.DataFrame(temp)
    return df

def load_data(args, remove = None):
    # imag1, _ques1, _answ1 = extract_data(args.train_text_file)
    traindf=extract_data(args.train_text_file)
    valdf=extract_data(args.valid_text_file)
    testdf=extract_data(args.test_text_file)

    if remove is not None:
        traindf = traindf[~traindf[1].isin(remove)].reset_index(drop=True)

    traindf[1] = traindf[1].apply(lambda x: os.path.join(args.train_image_file, x + '.jpg'))
    valdf[1] = valdf[1].apply(lambda x: os.path.join(args.valid_image_file, x + '.jpg'))
    testdf[1] = testdf[1].apply(lambda x: os.path.join(args.test_image_file, x + '.jpg'))
    # testdf['img_id'] = testdf['img_id'].apply(lambda x: os.path.join(args.data_dir, x + '.jpg'))

    # traindf['category'] = traindf['category'].str.lower()
    # valdf['category'] = valdf['category'].str.lower()
    # testdf['category'] = testdf['category'].str.lower()


    traindf[3] = traindf[3].str.lower()
    valdf[3] = valdf[3].str.lower()
    testdf[3] = testdf[3].str.lower()

    traindf = traindf.sample(frac = args.train_pct)
    valdf = valdf.sample(frac = args.valid_pct)
    testdf = testdf.sample(frac = args.test_pct)


    return traindf, valdf, testdf

traindf, valdf, testdf=load_data(args)

# VQA_MED
val_images, val_questions, val_answers = Utility.read_dataset(valdf,'Valid')
train_images, train_questions, train_answers = Utility.read_dataset(traindf,'Train')
test_images, test_questions = Utility.read_dataset(testdf,'Test')

# extract features from each photo in the directory

def extract_features(image_list):
        features = dict()
       # load the model
        model = VGG16(weights=args.pretrained)
       # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        c =1
       # extract features from each photo
        for filename in image_list:
              # load an image from file
                #filename = directory + '/' + name
                image = load_img(filename, target_size=(224, 224))
              # convert the image pixels to a numpy array
                image = img_to_array(image)
              # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
              # prepare the image for the VGG model
                image = preprocess_input(image)
              # get features
                feature = model.predict(image, verbose=0)
              # get image id
                image_id = filename.split('/')[-1].split('.')[0]
              # store feature
                features[image_id] = feature
                c+=1
        return features

'''
初次运行 要运行下面这段 生成pkl文件
'''
train_features = extract_features(train_images)

valid_features = extract_features(val_images)

test_features = extract_features(test_images)

import pdb;
# pdb.set_trace()
from pickle import load
def load_photo_features(filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k.split('/')[-1].split('.')[0]: all_features[k.split('/')[-1].split('.')[0]] for k in dataset}
        return features

#
lines = pd.DataFrame({'eng':train_questions, 'fr':train_answers})

eng = lines.eng.tolist() + val_questions.tolist()
fr = lines.fr.tolist() + val_answers.tolist()

lines = pd.DataFrame({'eng':eng, 'fr':fr})
lines.fr = lines.fr.apply(lambda x : 'START_ '+ x + ' _END')

import pdb;
# pdb.set_trace()
all_eng_words=set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

for eng in val_questions:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

for eng in test_questions:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_french_words=set()
for fr in lines.fr:
    for word in fr.split():
        if word not in all_french_words:
            all_french_words.add(word)

for eng in val_answers:
    for word in eng.split():
        if word not in all_eng_words:
            all_french_words.add(word)
# Answers
lenght_list=[]
for l in lines.fr:
    lenght_list.append(len(l.split(' ')))

# Questions
lenght_list=[]
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_french_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_french_words)
# del all_eng_words, all_french_words

input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])

encoder_input_data = np.zeros(
    (len(lines.eng), 29),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(lines.fr), 28),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(lines.fr), 28, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(lines.eng , lines.fr)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

# GLOVE_DIR = "glove"
embeddings_index = {}
f = open(args.glove,'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# import pdb; pdb.set_trace()

embedding_size = args.emb_size
embedding_matrix = np.zeros((len(input_token_index) , embedding_size))
for word, i in input_token_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix_2 = np.zeros((len(target_token_index) , embedding_size))
for word, i in target_token_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_2[i] = embedding_vector
        '''===============================Model Archi=============================='''
# Image Model
# feature extractor model

inputs1 = Input(shape=(4096,))
fe11 = Dense(2500, activation='relu')(inputs1)
fe2 = Dense(args.hidden_nodes, activation='relu')(fe11)

# Encoder model
encoder_inputs = Input(shape=(None,))
en_x=  Embedding(num_encoder_tokens, embedding_size, weights=[embedding_matrix], trainable=True)(encoder_inputs)
encoder = LSTM(args.hidden_nodes, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(en_x)

encoder_h = keras.layers.concatenate([state_h, fe2])
encoder_c = keras.layers.concatenate([state_c, fe2])
encoder_states = [encoder_h, encoder_c]

# Decoder model
decoder_inputs = Input(shape=(None,))
dex=  Embedding(num_decoder_tokens, embedding_size,weights=[embedding_matrix_2], trainable=True)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(args.hidden_nodes *2 , return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#model = Model([encoder_inputs, decoder_inputs, inputs1], decoder_outputs)
model = Model([encoder_inputs, decoder_inputs, inputs1], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

encoder_images = [i.split('/')[-1].split('.')[0] for i in train_images+val_images]
encoder_input_images = np.array([train_features[i][0] if i in train_features else valid_features[i][0] for i in encoder_images ])


encoder_model = Model([encoder_inputs, inputs1], encoder_states)


decoder_state_input_h = Input(shape=(args.hidden_nodes *2,))
decoder_state_input_c = Input(shape=(args.hidden_nodes *2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# encoder_model.save('save/test2_encoder.h5')
# decoder_model.save('save/test2_decoder.h5')
# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq, image_features):

    feature = np.array([image_features])
    states_value = encoder_model.predict([np.array([input_seq]), feature]) #encoder_model.predict(np.array([input_seq, feature]))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


encoder_images_val = [i.split('/')[-1].split('.')[0] for i in val_images]
encoder_input_images_val = np.array([valid_features[i][0] for i in encoder_images_val])
val_tokens = np.array([[input_token_index[i]
                        for i in val_ans.split()] for val_ans in val_questions])

encoder_images_test= [i.split('/')[-1].split('.')[0] for i in test_images]
encoder_input_images_test = np.array([test_features[i][0] for i in encoder_images_test])


def train():
    nltk_bleu=0.0
    for i in range(epochs):
        model.fit([encoder_input_data, decoder_input_data, encoder_input_images], decoder_target_data, batch_size=512,
                  epochs=1, validation_split=0,verbose=0)
        # if (i + 1) % 10 == 0:
        actual = []
        pred = []
        actual_nltk = []
        pred_nltk = []
        # output_valid = open("test_results/valid_out_"+str(i)+".txt", "w")
        for seq_index in range(len(val_tokens)):
            input_seq = val_tokens[seq_index]
            decoded_sentence = decode_sequence(input_seq, encoder_input_images_val[seq_index])

            ac = val_answers[seq_index].replace("START_", "")
            ac = ac.replace('_END', "").strip()

            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip()

            # output_valid.write(str(seq_index)+"\t"+val_questions[seq_index]+"\t"+ac+"\t"+pr+"\n")
            actual.append(ac)
            pred.append(pr)
            actual_nltk.append(ac.strip())
            pred_nltk.append(pr.strip())
            # output_valid.close()

        src_new = [[i.strip().split()] for i in actual_nltk]
        trg_new = [i.strip().split() for i in pred_nltk]

        nltk_bleu = nltk.translate.bleu_score.corpus_bleu(src_new, trg_new,weights=(0, 1, 0, 0))

    # project_path=os.path.abspath(os.path.dirname(__file__))
    model.save(args.save_dir+'/'+args.run_name+'.h5')
    print("{bleu}".format(bleu=round(nltk_bleu, 5)))
# writer.close()
# test_output.close()
def predict(question,imagpath):
    # imagpath='dataset/VQAMed2018Test/VQAMed2018Test-Images/synpic20053.jpg'
    imag_id=imagpath.split('/')[-1].split('.')[0]
    imag_feature=extract_features([imagpath])
    imag_feature=imag_feature[imag_id]
    feature=np.arange(imag_feature.shape[1]).astype(np.float32)
    for i in range(feature.shape[0]):
        feature[i]=float(imag_feature[:,i])
    test_tokens = np.array([input_token_index[i] for i in question.split()])
    input_seq = test_tokens
    decoded_sentence = decode_sequence(input_seq, feature)

    pr = decoded_sentence.replace("START_", "")
    pr = pr.replace('_END', "").strip()
    print(pr)

if __name__=="__main__":
    if args.mode=='train':
        train()
    else:
        predict(args.question,args.imag)