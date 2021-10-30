from __future__ import print_function
import os
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
batch_size = 256 # Batch size for training.
epochs = 1000# Number of epochs to train for.


# VQA_MED
val_images, val_questions, val_answers = Utility.read_dataset("Valid")
train_images, train_questions, train_answers = Utility.read_dataset("Train")
test_images, test_questions = Utility.read_dataset("Test")

# extract features from each photo in the directory
features = dict()
def extract_features(image_list):
       # load the model
        model = VGG16()
       # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
       # summarize
       #  print(model.summary())
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
                # print(str(c), '>%s' % filename, "\t", image_id)
                c+=1
        return features
#print('extract train images features')
#train_features = extract_features(train_images)
#dump(train_features, open('train_features.pkl', 'wb'))

#extract_features(train_images)
#print('extract valid images features')
#valid_features = extract_features(val_images)
#dump(valid_features, open('valid_features.pkl', 'wb'))

#print('extract test images features')
#test_features = extract_features(test_images)
#dump(test_features, open('test_features.pkl', 'wb'))
#print('Extracted Features: %d' % len(features))
# load photo features
#import pdb; pdb.set_trace()
from pickle import load
def load_photo_features(filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k.split('/')[-1].split('.')[0]: all_features[k.split('/')[-1].split('.')[0]] for k in dataset}
        return features

print("load train features")
train_features = load_photo_features('train_features.pkl', train_images)
print("load val features")
valid_features = load_photo_features('valid_features.pkl', val_images)
print("load test features")
test_features = load_photo_features('test_features.pkl', test_images)

lines = pd.DataFrame({'eng':train_questions, 'fr':train_answers})

eng = lines.eng.tolist() + val_questions.tolist()
fr = lines.fr.tolist() + val_questions.tolist()

lines = pd.DataFrame({'eng':eng, 'fr':fr})
lines.fr = lines.fr.apply(lambda x : 'START_ '+ x + ' _END')

import pdb; pdb.set_trace()
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

print ("Load pretrained embeddings ...")
GLOVE_DIR = "glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# import pdb; pdb.set_trace()
emb_size = 300
hidden_nodes = 1024
embedding_size = emb_size
embedding_matrix = np.zeros((len(input_token_index) , emb_size))
for word, i in input_token_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix_2 = np.zeros((len(target_token_index) , emb_size))
for word, i in target_token_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_2[i] = embedding_vector
# Image Model
# feature extractor model

inputs1 = Input(shape=(4096,))
fe11 = Dense(2500, activation='relu')(inputs1)
fe2 = Dense(hidden_nodes, activation='relu')(fe11)

# ----------------------------------------------------------
# vis_model = Sequential()
# vis_model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
# vis_model.add(Convolution2D(64, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(64, 3, 3, activation='relu'))
# vis_model.add(MaxPooling2D((2,2), strides=(2,2)))
# #
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(128, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(128, 3, 3, activation='relu'))
# vis_model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(256, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(256, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(256, 3, 3, activation='relu'))
# vis_model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(512, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(512, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(512, 3, 3, activation='relu'))
# vis_model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(512, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(512, 3, 3, activation='relu'))
# vis_model.add(ZeroPadding2D((1,1)))
# vis_model.add(Convolution2D(512, 3, 3, activation='relu'))
# vis_model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# vision_model = Sequential()
# vision_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
# #vision_model.add(Conv2D(32, (3, 3), activation='relu'))
# vision_model.add(MaxPooling2D((2, 2)))
# # `vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# `vision_model.add(Conv2D(128, (3, 3), activation='relu'))
# `vision_model.add(MaxPooling2D((2, 2)))
# `vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# `vision_model.add(Conv2D(256, (3, 3), activation='relu'))
# `vision_model.add(Conv2D(256, (3, 3), activation='relu'))
# `vision_model.add(MaxPooling2D((2, 2)))
# `vision_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# `vision_model.add(Conv2D(512, (3, 3), activation='relu'))
# `vision_model.add(Conv2D(512, (3, 3), activation='relu'))
# `vision_model.add(MaxPooling2D((2, 2)))
# `vision_model.add(Flatten())
# `vision_model.add(Dense(4096, activation='relu'))
#vision_model.add(Flatten())

#vision_model.add(Dense(1024, activation='relu'))
#image_input = Input(shape=(224, 224, 3))
#inputs1 = Input(shape=(224, 224, 3))
#encoded_image = vision_model(inputs1)

#fe1 = Dropout(0.5)(encoded_image)
#fe2 = Dense(hidden_nodes, activation='relu')(encoded_image)
#vis_model.add(Dropout(0.5))
#vis_model.add(Dense(4096, activation='relu'))
# ----------------------------------------------------------


# Encoder model
encoder_inputs = Input(shape=(None,))
en_x=  Embedding(num_encoder_tokens, embedding_size, weights=[embedding_matrix], trainable=True)(encoder_inputs)
encoder = LSTM(hidden_nodes, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
# encoder_states = [state_h, state_c]

#import pdb; pdb.set_trace()
# Combine CNN and RNN to create the final model
#merged2 = keras.layers.concatenate([encoder_states, encoded_image])

encoder_h = keras.layers.concatenate([state_h, fe2])
encoder_c = keras.layers.concatenate([state_c, fe2])
encoder_states = [encoder_h, encoder_c]

# Decoder model
decoder_inputs = Input(shape=(None,))
dex=  Embedding(num_decoder_tokens, embedding_size,weights=[embedding_matrix_2], trainable=True)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(hidden_nodes *2 , return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#model = Model([encoder_inputs, decoder_inputs, inputs1], decoder_outputs)
model = Model([encoder_inputs, decoder_inputs, inputs1], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

encoder_images = [i.split('/')[-1].split('.')[0] for i in train_images]
encoder_input_images = np.array([train_features[i][0] if i in train_features else valid_features[i][0] for i in encoder_images ])

# encoder_images = [load_img(i, target_size=(224, 224)) for i in train_images]
# encoder_input_images = np.array([img_to_array(i) for i in encoder_images])
#
# encoder_images_val = [load_img(i, target_size=(224, 224)) for i in val_images]
# encoder_input_images_val = np.array([img_to_array(i) for i in encoder_images_val])





#model.fit([encoder_input_data, decoder_input_data, encoder_input_images], decoder_target_data, batch_size=128, epochs=3,validation_split=0)
##import pdb; pdb.set_trace()



# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

encoder_model = Model([encoder_inputs, inputs1], encoder_states)


decoder_state_input_h = Input(shape=(hidden_nodes *2,))
decoder_state_input_c = Input(shape=(hidden_nodes *2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq, image_features):
    # Encode the input as state vectors.
#    import pdb; pdb.set_trace()
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


from bleu import moses_multi_bleu
writer = open("results.txt", "a")
#test_output= open("output.txt", "a")


encoder_images_val = [i.split('/')[-1].split('.')[0] for i in val_images]
encoder_input_images_val = np.array([valid_features[i][0] for i in encoder_images_val])
val_tokens = np.array([[input_token_index[i]
                        for i in val_ans.split()] for val_ans in val_questions])

encoder_images_test= [i.split('/')[-1].split('.')[0] for i in test_images]
encoder_input_images_test = np.array([test_features[i][0] for i in encoder_images_test])

# encoder_images_test = [load_img(i, target_size=(224, 224)) for i in test_images]
# encoder_input_images_test = np.array([img_to_array(i) for i in encoder_images_test])
test_tokens = np.array([[input_token_index[i] for i in test_ans.split()]
                        for test_ans in test_questions])

for i in range(epochs):
    print("epoch", str(i), "out of",str(epochs) )
    model.fit([encoder_input_data, decoder_input_data, encoder_input_images], decoder_target_data, batch_size=512, epochs=1,validation_split=0)
    if (i+1)%10==0:

        writer.write('\n------ epoch '+str(i)+" ------\n")
        print ("Validating ...")
        actual = []
        pred = []
        actual_nltk = []
        pred_nltk = []
        output_valid = open("test_results/valid_out_"+str(i)+".txt", "w")
        for seq_index in range(len(val_tokens)):
            input_seq = val_tokens[seq_index]
            decoded_sentence = decode_sequence(input_seq, encoder_input_images_val[seq_index])

            ac = val_answers[seq_index].replace("START_", "")
            ac = ac.replace('_END', "").strip()

            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip()
            if seq_index<=20:
               print('-')
               print('Input sentence:', val_questions[seq_index])
               print('Actual sentence:', ac)
               print('Decoded sentence:', pr)
            # test_output.write('Input sentence: '+ val_questions[seq_index]+"\n")
            # test_output.write('Actual sentence: '+ ac+"\n")
            # test_output.write('Decoded sentence: '+ pr+"\n")

            output_valid.write(str(seq_index)+"\t"+val_questions[seq_index]+"\t"+ac+"\t"+pr+"\n")
            actual.append(ac)
            pred.append(pr)
            actual_nltk.append(ac.strip())
            pred_nltk.append(pr.strip())
        output_valid.close()

        src_new = [[i.strip().split()] for i in actual_nltk]
        trg_new = [i.strip().split() for i in pred_nltk]

        nltk_bleu = nltk.translate.bleu_score.corpus_bleu(src_new, trg_new)
        bleus = moses_multi_bleu(actual, pred)

        writer.write("Moses Test Bleu: "+ str(bleus)+"\n")
        print("Moses Test Bleu:", str(bleus))
        writer.write("NLTK Test Bleu: "+ str(nltk_bleu)+"\n")
        print("NLTK Test Bleu:", str(nltk_bleu))


        actual = []
        pred = []
        actual_nltk = []
        pred_nltk = []
        for seq_index in range(500):
            input_seq = encoder_input_data[seq_index]
            decoded_sentence = decode_sequence(input_seq, encoder_input_images[seq_index])

            ac = lines.fr[seq_index].replace("START_", "")
            ac = ac.replace('_END', "").strip()

            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip()
            if seq_index<=20:

                print('-')
                print('Input sentence:', lines.eng[seq_index])
                print('Actual sentence:', ac)
                print('Decoded sentence:', pr)

            actual.append(ac)
            pred.append(pr)
            actual_nltk.append(ac.strip())
            pred_nltk.append(pr.strip())

        src_new = [[i.strip().split()] for i in actual_nltk]
        trg_new = [i.strip().split() for i in pred_nltk]

        nltk_bleu = nltk.translate.bleu_score.corpus_bleu(src_new, trg_new)
        bleus = moses_multi_bleu(actual, pred)
        writer.write("Moses Train Bleu: "+ str(bleus)+"\n")
        writer.write("NLTK Train Bleu: "+ str(nltk_bleu)+"\n")
        writer.flush()
        print("Moses Train Bleu:", str(bleus))
        print("NLTK Train Bleu:", str(nltk_bleu))

        print ("\nTesting...")
        output = open("test_results/out_"+str(i)+".txt", "w")
        for seq_index in range(len(test_tokens)):
            input_seq = test_tokens[seq_index]
            decoded_sentence = decode_sequence(input_seq, encoder_input_images_test[seq_index])


            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip()
            if seq_index<=20:
               print('-')
               print('Input sentence:', test_questions[seq_index])
               print('Decoded sentence:', pr)
            output.write(str(seq_index)+"\t"+test_questions[seq_index]+"\t"+pr+"\n")
        output.close()

writer.close()
# test_output.close()