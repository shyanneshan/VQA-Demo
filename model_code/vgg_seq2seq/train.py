from __future__ import print_function

import argparse
import os
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from data_utils import Utility
import pandas as pd
import numpy as np
import os
import nltk


# batch_size = 256 # Batch size for training.
# epochs = 1000# Number of epochs to train for.


# VQA_MED
# val_images, val_questions, val_answers = Utility.read_val_dataset("Valid")
# train_images, train_questions, train_answers = Utility.read_train_dataset("Train")
# test_images, test_questions = Utility.read_test_dataset("Test")
#
# # extract features from each photo in the directory
# features = dict()
def extract_features(image_list):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    # print(model.summary())
    c = 1
    # extract features from each photo
    for filename in image_list:
        # load an image from file
        # filename = directory + '/' + name
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
        c += 1
    return features


from pickle import load


def load_photo_features(filename, dataset):
    # load all features
    # 这里的feature  是extract_features() 获得的
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k.split('/')[-1].split('.')[0]: all_features[k.split('/')[-1].split('.')[0]] for k in dataset}
    return features


# print("load train features")
# train_features = load_photo_features('train_features.pkl', train_images)
# print("load val features")
# valid_features = load_photo_features('valid_features.pkl', val_images)
# print("load test features")
# test_features = load_photo_features('test_features.pkl', test_images)


# lines = pd.DataFrame({'eng':train_questions, 'fr':train_answers})
#
# eng = lines.eng.tolist() + val_questions.tolist()
# fr = lines.fr.tolist() + val_questions.tolist()
#
# lines = pd.DataFrame({'eng':eng, 'fr':fr})
# lines.fr = lines.fr.apply(lambda x : 'START_ '+ x + ' _END')

def transform2pkl(args,train_images, val_images, test_images):
    train_features = extract_features(train_images)
    val_features = extract_features(val_images)
    test_features = extract_features(test_images)
    with open(os.path.join(args.data_dir,"train_features.pkl"), "wb") as train:
        pickle.dump(train_features, train)
    with open(os.path.join(args.data_dir,"val_features.pkl"), "wb") as val:
        pickle.dump(val_features, val)
    with open(os.path.join(args.data_dir,"test_features.pkl"), "wb") as test:
        pickle.dump(test_features, test)


# import pdb; pdb.set_trace()
# all_eng_words=set()
# for eng in lines.eng:
#     for word in eng.split():
#         if word not in all_eng_words:
#             all_eng_words.add(word)
#
# for eng in val_questions:
#     for word in eng.split():
#         if word not in all_eng_words:
#             all_eng_words.add(word)
#
# for eng in test_questions:
#     for word in eng.split():
#         if word not in all_eng_words:
#             all_eng_words.add(word)
def getQuesSet(lines, val_questions, test_questions):  # 从train val test三个数据集中获得词汇集合
    all_eng_words = set()
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

    return all_eng_words


# all_french_words=set()
# for fr in lines.fr:
#     for word in fr.split():
#         if word not in all_french_words:
#             all_french_words.add(word)
#
# for eng in val_answers:
#     for word in eng.split():
#         if word not in all_eng_words:
#             all_french_words.add(word)

def getAnsSet(lines, val_answers):  # 获得train val答案的词汇
    all_french_words = set()
    for fr in lines.fr:
        for word in fr.split():
            if word not in all_french_words:
                all_french_words.add(word)

    for eng in val_answers:
        for word in eng.split():
            if word not in all_french_words:
                all_french_words.add(word)
    return all_french_words


# Answers #没用上 注释掉了
# lenght_list=[]
# for l in lines.fr:
#     lenght_list.append(len(l.split(' ')))
#
# # Questions
# lenght_list=[]
# for l in lines.eng:
#     lenght_list.append(len(l.split(' ')))

# input_words = sorted(list(all_eng_words))
# target_words = sorted(list(all_french_words))
# num_encoder_tokens = len(all_eng_words)
# num_decoder_tokens = len(all_french_words)
# # del all_eng_words, all_french_words
#
# input_token_index = dict(
#     [(word, i) for i, word in enumerate(input_words)])
# target_token_index = dict(
#     [(word, i) for i, word in enumerate(target_words)])

def process(lines, num_decoder_tokens, input_token_index, target_token_index):
    encoder_input_data = np.zeros(
        (len(lines.eng), 29),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(lines.fr), 28),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(lines.fr), 28, num_decoder_tokens),
        dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(lines.eng, lines.fr)):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[i, t] = input_token_index[word]
        for t, word in enumerate(target_text.split()):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t] = target_token_index[word]
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[word]] = 1.
    return encoder_input_data, decoder_input_data, decoder_target_data


def load_glove(path):
    # print("Load pretrained embeddings ...")
    GLOVE_DIR = "/data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq/glove"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, path), 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def embedding_vector(embeddings_index, input_token_index, target_token_index):
    # print('Found %s word vectors.' % len(embeddings_index))
    # import pdb; pdb.set_trace()
    emb_size = 300
    hidden_nodes = 1024
    embedding_size = emb_size
    embedding_matrix = np.zeros((len(input_token_index), emb_size))
    for word, i in input_token_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_matrix_2 = np.zeros((len(target_token_index), emb_size))
    for word, i in target_token_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_2[i] = embedding_vector
    return embedding_matrix, embedding_matrix_2


class EncoderModel():
    def __init__(self, hidden_nodes, num_encoder_tokens, embedding_size, embedding_matrix):
        self.hidden_nodes = hidden_nodes
        self.num_encoder_tokens = num_encoder_tokens
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix

    def createModel(self):
        # Image Model
        # feature extractor model
        inputs1 = Input(shape=(1000,))
        fe11 = Dense(2500, activation='relu')(inputs1)
        fe2 = Dense(self.hidden_nodes, activation='relu')(fe11)

        # Encoder model
        encoder_inputs = Input(shape=(None,))
        en_x = Embedding(self.num_encoder_tokens, self.embedding_size, weights=[self.embedding_matrix], trainable=True)(
            encoder_inputs)
        encoder = LSTM(self.hidden_nodes, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(en_x)

        encoder_h = keras.layers.concatenate([state_h, fe2])
        encoder_c = keras.layers.concatenate([state_c, fe2])
        encoder_states = [encoder_h, encoder_c]

        encoder_model = Model([encoder_inputs, inputs1], encoder_states)

        return encoder_model


class DecoderModel():
    def __init__(self, lines, num_decoder_tokens,
                 hidden_nodes, num_encoder_tokens, embedding_size, embedding_matrix, embedding_matrix_2):
        # self.lines = lines
        # self.num_decoder_tokens = num_decoder_tokens
        self.encoder_input_data = np.zeros(
            (len(lines.eng), 29),
            dtype='float32')
        self.decoder_input_data = np.zeros(
            (len(lines.fr), 28),
            dtype='float32')
        self.decoder_target_data = np.zeros(
            (len(lines.fr), 28, num_decoder_tokens),
            dtype='float32')
        self.hidden_nodes = hidden_nodes
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.embedding_matrix_2 = embedding_matrix_2

    def createModel(self):
        # Image Model
        # feature extractor model
        inputs1 = Input(shape=(1000,))
        fe11 = Dense(2500, activation='relu')(inputs1)
        fe2 = Dense(self.hidden_nodes, activation='relu')(fe11)

        # Encoder model
        encoder_inputs = Input(shape=(None,))
        en_x = Embedding(num_encoder_tokens, embedding_size, weights=[embedding_matrix], trainable=True)(encoder_inputs)
        encoder = LSTM(hidden_nodes, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(en_x)

        encoder_h = keras.layers.concatenate([state_h, fe2])
        encoder_c = keras.layers.concatenate([state_c, fe2])
        encoder_states = [encoder_h, encoder_c]

        # Decoder model
        decoder_inputs = Input(shape=(None,))
        dex = Embedding(self.num_decoder_tokens, self.embedding_size, weights=[self.embedding_matrix_2], trainable=True)
        final_dex = dex(decoder_inputs)
        decoder_lstm = LSTM(self.hidden_nodes * 2, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(final_dex,
                                             initial_state=encoder_states)

        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        # decoder_outputs = decoder_dense(decoder_outputs)

        decoder_state_input_h = Input(shape=(hidden_nodes * 2,))
        decoder_state_input_c = Input(shape=(hidden_nodes * 2,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        final_dex2 = dex(decoder_inputs)

        decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)

        return decoder_model


class Seq2SeqModel():
    def __init__(self, hidden_nodes, num_encoder_tokens, num_decoder_tokens,
                 embedding_size, embedding_matrix, embedding_matrix_2):
        self.hidden_nodes = hidden_nodes
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.embedding_matrix_2 = embedding_matrix_2
        # self.encoder=encoder
        # self.decoder=decoder

    # def getEncoder(self):
    #     return self.encoder
    #
    # def getDecoder(self):
    #     return self.decoder

    def createModel(self):
        # Image Model
        # feature extractor model
        inputs1 = Input(shape=(1000,))
        fe11 = Dense(2500, activation='relu')(inputs1)
        fe2 = Dense(self.hidden_nodes, activation='relu')(fe11)

        # Encoder model
        encoder_inputs = Input(shape=(None,))
        en_x = Embedding(self.num_encoder_tokens, self.embedding_size, weights=[self.embedding_matrix], trainable=True)(
            encoder_inputs)
        encoder = LSTM(self.hidden_nodes, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(en_x)

        encoder_h = keras.layers.concatenate([state_h, fe2])
        encoder_c = keras.layers.concatenate([state_c, fe2])
        encoder_states = [encoder_h, encoder_c]

        # Decoder model
        decoder_inputs = Input(shape=(None,))
        dex = Embedding(self.num_decoder_tokens, self.embedding_size, weights=[self.embedding_matrix_2], trainable=True)
        final_dex = dex(decoder_inputs)
        decoder_lstm = LSTM(self.hidden_nodes * 2, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(final_dex,
                                             initial_state=encoder_states)

        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs, inputs1], decoder_outputs)

        return model


def decode_sequence(encoder_model, decoder_model, reverse_target_char_index, input_seq, image_features):
    # Encode the input as state vectors.
    #    import pdb; pdb.set_trace()
    feature = np.array([image_features])
    states_value = encoder_model.predict(
        [np.array([input_seq]), feature])  # encoder_model.predict(np.array([input_seq, feature]))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
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
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
                len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# from bleu import moses_multi_bleu
# writer = open("results.txt", "a")
# test_output= open("output.txt", "a")


# encoder_images_val = [i.split('/')[-1].split('.')[0] for i in val_images]
# encoder_input_images_val = np.array([valid_features[i][0] for i in encoder_images_val])
# val_tokens = np.array([[input_token_index[i]
#                         for i in val_ans.split()] for val_ans in val_questions])
#
# encoder_images_test= [i.split('/')[-1].split('.')[0] for i in test_images]
# encoder_input_images_test = np.array([test_features[i][0] for i in encoder_images_test])
#
# # encoder_images_test = [load_img(i, target_size=(224, 224)) for i in test_images]
# # encoder_input_images_test = np.array([img_to_array(i) for i in encoder_images_test])
# test_tokens = np.array([[input_token_index[i] for i in test_ans.split()]
#                         for test_ans in test_questions])


def train(args,epochs, encoder_model, decoder_model, model,
          encoder_input_images, encoder_input_images_val,
          decoder_input_data, decoder_target_data, val_questions, val_answers, val_tokens,
          test_questions, test_answers, test_tokens,encoder_input_images_test):
    writer = open("results.txt", "a")

    for i in range(int(args.epochs)):
        # print("epoch", str(i), "out of", str(epochs))
        # encoder_model.fit()
        model.fit([encoder_input_data, decoder_input_data, encoder_input_images], decoder_target_data, batch_size=args.batch_size,verbose=0)#epochs=1, validation_split=0
        if (i + 1) % 10 == 0:

            writer.write('\n------ epoch ' + str(i) + " ------\n")
            # print("Validating ...")
            actual = []
            pred = []
            actual_nltk = []
            pred_nltk = []
            output_valid = open("valid_out_" + str(i) + ".txt", "w")
            for seq_index in range(len(val_tokens)):
                input_seq = val_tokens[seq_index]
                decoded_sentence = decode_sequence(encoder_model, decoder_model, reverse_target_char_index, input_seq,
                                                   encoder_input_images_val[seq_index])

                ac = val_answers[seq_index].replace("START_", "")
                ac = ac.replace('_END', "").strip()

                pr = decoded_sentence.replace("START_", "")
                pr = pr.replace('_END', "").strip()
                # if seq_index <= 20:
                #     print('-')
                #     print('Input sentence:', val_questions[seq_index])
                #     print('Actual sentence:', ac)
                #     print('Decoded sentence:', pr)
                # test_output.write('Input sentence: '+ val_questions[seq_index]+"\n")
                # test_output.write('Actual sentence: '+ ac+"\n")
                # test_output.write('Decoded sentence: '+ pr+"\n")

                output_valid.write(str(seq_index) + "\t" + val_questions[seq_index] + "\t" + ac + "\t" + pr + "\n")
                actual.append(ac)
                pred.append(pr)
                actual_nltk.append(ac.strip())
                pred_nltk.append(pr.strip())
            output_valid.close()

            src_new = [[i.strip().split()] for i in actual_nltk]
            trg_new = [i.strip().split() for i in pred_nltk]

            # nltk_bleu = nltk.translate.bleu_score.corpus_bleu(src_new, trg_new)

    writer.close()
    # TestSet bleu
    actual = []
    pred = []
    actual_nltk = []
    pred_nltk = []
    # output_valid = open("valid_out_" + str(i) + ".txt", "w")
    for seq_index in range(len(test_tokens)):
        input_seq = test_tokens[seq_index]
        decoded_sentence = decode_sequence(encoder_model, decoder_model, reverse_target_char_index, input_seq,
                                           encoder_input_images_test[seq_index])

        ac = test_answers[seq_index].replace("START_", "")
        ac = ac.replace('_END', "").strip()

        pr = decoded_sentence.replace("START_", "")
        pr = pr.replace('_END', "").strip()
        # output_valid.write(str(seq_index) + "\t" + test_questions[seq_index] + "\t" + ac + "\t" + pr + "\n")
        actual.append(ac)
        pred.append(pr)
        actual_nltk.append(ac.strip())
        pred_nltk.append(pr.strip())
    # output_valid.close()

    src_new = [[i.strip().split()] for i in actual_nltk]
    trg_new = [i.strip().split() for i in pred_nltk]

    nltk_bleu = nltk.translate.bleu_score.corpus_bleu(src_new, trg_new)
    print(nltk_bleu)
    # return nltk_bleu

def predict(test_tokens, test_questions,
            encoder_model, decoder_model,
            reverse_target_char_index,
            encoder_input_images_test):
    print("\nTesting...")
    # output = open("test_results/out_" + str(i) + ".txt", "w")
    for seq_index in range(len(test_tokens)):
        input_seq = test_tokens[seq_index]
        decoded_sentence = decode_sequence(encoder_model, decoder_model, reverse_target_char_index, input_seq,
                                           encoder_input_images_test[seq_index])

        pr = decoded_sentence.replace("START_", "")
        pr = pr.replace('_END', "").strip()
        if seq_index <= 20:
            print('-')
            print('Input sentence:', test_questions[seq_index])
            print('Decoded sentence:', pr)
        # output.write(str(seq_index) + "\t" + test_questions[seq_index] + "\t" + pr + "\n")
    # output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default="dataset",
                        help='the position of dataset trained on.')
    parser.add_argument('--model_dir', default="dataset/dataset.pt",
                        help='the position of dataset trained on.')
    parser.add_argument('--epochs',type=int, default=10,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch-size', type=int,default=16,
                        help='Batch size for training.')
    parser.add_argument('--traintxt', default="dataset/train.txt",
                        help='Questions Answers ImgId for train.')
    parser.add_argument('--trainimg', default="dataset/train",
                        help='Image Dictionary for train.')
    parser.add_argument('--valtxt', default="dataset/val.txt",
                        help='Questions Answers ImgId for Val.')
    parser.add_argument('--valimg', default="dataset/val",
                        help='Image Dictionary for Val.')
    parser.add_argument('--testtxt', default="dataset/test.txt",
                        help='Questions Answers ImgId for Test.')
    parser.add_argument('--testimg', default="dataset/test",
                        help='Image Dictionary for Test.')
    # Ablations.
    parser.add_argument('--no-image-recon', action='store_true', default=True,
                        help='Does not try to reconstruct image.')
    parser.add_argument('--no-question-recon', action='store_true', default=True,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no-caption-recon', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no-category-space', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')

    args = parser.parse_args()

    val_images, val_questions, val_answers = Utility.read_val_dataset(args.valtxt, args.valimg)
    train_images, train_questions, train_answers = Utility.read_train_dataset(args.traintxt, args.trainimg)
    test_images, test_questions, test_answers = Utility.read_test_dataset(args.testtxt, args.testimg)

    features = dict()

    # 将图片处理为pkl形式
    transform2pkl(args,train_images, val_images, test_images)

    # print("load train features")
    train_features = load_photo_features(os.path.join(args.data_dir,"train_features.pkl"), train_images)
    # print("load val features")
    valid_features = load_photo_features(os.path.join(args.data_dir,"val_features.pkl"), val_images)
    # print("load test features")
    test_features = load_photo_features(os.path.join(args.data_dir,"test_features.pkl"), test_images)

    # 将questions和answers处理成表格形式，answers加入前缀和后缀
    lines = pd.DataFrame({'eng': train_questions, 'fr': train_answers})

    eng = lines.eng + val_questions
    fr = lines.fr + val_answers
    # eng = lines.eng.tolist() + val_questions.tolist()
    # fr = lines.fr.tolist() + val_questions.tolist()

    lines = pd.DataFrame({'eng': eng, 'fr': fr})
    lines.fr = lines.fr.apply(lambda x: 'START_ ' + x + ' _END')

    # 获得questions和answers的词汇集合
    all_eng_words = getQuesSet(lines, val_questions, test_questions)
    all_french_words = getAnsSet(lines, val_answers)

    # 处理词汇 排序成输入输出、词汇总个数是tokens长度（防止超出长度，就是有点费空间
    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_french_words))
    num_encoder_tokens = len(all_eng_words)
    num_decoder_tokens = len(all_french_words)
    # del all_eng_words, all_french_words

    # 形式为（词汇，id）的输入和输出index词典
    input_token_index = dict(
        [(word, i) for i, word in enumerate(input_words)])
    target_token_index = dict(
        [(word, i) for i, word in enumerate(target_words)])
    # 设定参数
    emb_size = 300
    hidden_nodes = 1024
    embedding_size = emb_size

    encoder_input_data, decoder_input_data, decoder_target_data = process(lines, num_decoder_tokens, input_token_index,
                                                                          target_token_index)
    # 加载glove
    embeddings_index = load_glove('glove.6B.300d.txt')
    # 将input和target的内容转为向量格式
    embedding_matrix, embedding_matrix_2 = embedding_vector(embeddings_index, input_token_index, target_token_index)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    encoder_images = [i.split('/')[-1].split('.')[0] for i in train_images]
    encoder_input_images = np.array(
        [train_features[i][0] if i in train_features else valid_features[i][0] for i in encoder_images])

    encoder_images_val = [i.split('/')[-1].split('.')[0] for i in val_images]
    encoder_input_images_val = np.array([valid_features[i][0] for i in encoder_images_val])

    val_tokens = np.array([[input_token_index[i]
                            for i in val_ans.split()] for val_ans in val_questions])

    encoder_images_test = [i.split('/')[-1].split('.')[0] for i in test_images]
    encoder_input_images_test = np.array([test_features[i][0] for i in encoder_images_test])

    # encoder_images_test = [load_img(i, target_size=(224, 224)) for i in test_images]
    # encoder_input_images_test = np.array([img_to_array(i) for i in encoder_images_test])
    test_tokens = np.array([[input_token_index[i] for i in test_ans.split()]
                            for test_ans in test_questions])


    encoder_model = EncoderModel(hidden_nodes, num_encoder_tokens, embedding_size, embedding_matrix).createModel()
    decoder_model = DecoderModel(lines, num_decoder_tokens,
                                 hidden_nodes, num_encoder_tokens, embedding_size, embedding_matrix,embedding_matrix_2)\
                                .createModel()
    model = Seq2SeqModel(hidden_nodes, num_encoder_tokens, num_decoder_tokens,
                         embedding_size, embedding_matrix, embedding_matrix_2).createModel()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    train(args,args.epochs, encoder_model, decoder_model, model,
          encoder_input_images, encoder_input_images_val,
          decoder_input_data, decoder_target_data, val_questions, val_answers, val_tokens,
          test_questions, test_answers, test_tokens,encoder_input_images_test)
    # if
    model.save_weights(args.model_dir)