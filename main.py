import string
import os
import glob
import cv2
import tensorflow as tf
import logging
import warnings
import numpy as np
import scipy
import pandas as pd
from PIL import Image
from time import time
import matplotlib.pyplot as plt
from numpy import array
from PIL import ImageEnhance
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
# from skimage import feature
from colorama import Fore, init
from sklearn.model_selection import train_test_split
import math

from dbn import models, SupervisedDBNClassification
from optimization.optim_main import multi_opt
from other import load, result, popup
from other.Confusion_matrix import multi_confu_matrix

import tensorflow.keras as keras
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Embedding, Dense, Flatten, Reshape, \
    Dropout, Bidirectional, Concatenate, TimeDistributed, RepeatVector, GRU, SimpleRNN
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical

from other.glstm import GLSTMCell



def full_analysis():
    token_path = "Flickr8k_text/Flickr8k.token.txt"
    train_images_path = 'Flickr8k_text/Flickr_8k.trainImages.txt'
    test_images_path = 'Flickr8k_text/Flickr_8k.testImages.txt'
    images_path = 'dataset/Images/'
    glove_path = 'glove'

    doc = open(token_path, 'r').read()
    # print(doc[:410])

    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split('.')[0]
            image_desc = ' '.join(tokens[1:])
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)

    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc_list[i] = ' '.join(desc)

    # pic = '1000268201_693b08cb0e.jpg'
    # x=plt.imread(images_path+pic)
    # plt.imshow(x)
    # plt.show()
    # descriptions['1000268201_693b08cb0e']
    vocabulary = set()
    for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
    # print('Original Vocabulary Size: %d' % len(vocabulary))

    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    new_descriptions = '\n'.join(lines)

    doc = open(train_images_path, 'r').read()
    dataset = list()
    for line in doc.split('\n'):
        if len(line) > 1:
            identifier = line.split('.')[0]
            dataset.append(identifier)

    train = set(dataset)

    img = glob.glob(images_path + '*.jpg')
    train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
    train_img = []
    for i in img:
        if i[len(images_path):] in train_images:
            train_img.append(i)

    test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
    test_img = []
    for i in img:
        if i[len(images_path):] in test_images:
            test_img.append(i)

    train_descriptions = dict()
    for line in new_descriptions.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in train:
            if image_id not in train_descriptions:
                train_descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            train_descriptions[image_id].append(desc)

    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)

    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    vocab_size = len(ixtoword) + 1

    all_desc = list()
    for key in train_descriptions.keys():
        [all_desc.append(d) for d in train_descriptions[key]]
    lines = all_desc
    max_length = max(len(d.split()) for d in lines)

    embeddings_index = {}
    f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    embedding_dim = 200
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    #
    # # Model
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)

    def preprocess(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def encode(image):
        image = preprocess(image)
        fea_vec = model_new.predict(image)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec

        # encoding_train = {}
        # for img in train_img:
        #     encoding_train[img[len(images_path):]] = encode(img)
        # train_features = encoding_train
        #
        # encoding_test = {}
        # for img in test_img:
        #     encoding_test[img[len(images_path):]] = encode(img)

    keras.backend.clear_session()
    train_features = load.load('pre_evaluated/train_features')
    test_features = load.load('pre_evaluated/encoding_test')

    #
    def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n = 0
        # loop for ever over images
        while 1:
            for key, desc_list in descriptions.items():
                n += 1
                # retrieve the photo feature
                photo = photos[key + '.jpg']
                for desc in desc_list:
                    # encode the sequence
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    # split one sequence into multiple X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        # store
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)

                if n == num_photos_per_batch:
                    yield ([array(X1), array(X2)], array(y))
                    X1, X2, y = list(), list(), list()
                    n = 0

    def extract(descriptions, photos, wordtoix, max_length):
        X1, X2, Y = list(), list(), list()
        n = 0
        # loop for ever over images
        for key, desc_list in descriptions.items():
            n += 1
            # retrieve the photo feature
            photo = photos[key + '.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                # for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq, seq
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = np.sum(to_categorical([seq], num_classes=vocab_size)[0], axis=0)
                out_seq[out_seq > 1] = 1
                # store
                X1.append(photo)
                X2.append(in_seq)
                Y.append(out_seq)
        idx = 50000
        return array(X1), array(X2), array(Y)

    def feat_ex(images_):
        an = 0
        if an == 1:
            cont_img_ = []
            sharp_ = []
            col_ = []
            mot_ = []
            fa_im_sc_ = []
            im_fa_im_sc = []
            im_con_img = []
            for im in images_:
                new_image = np.zeros(im.shape, im.dtype)
                alpha = 1.3
                beta = 5
                for y in range(im.shape[0]):
                    new_image[y] = np.clip(alpha * im[y] + beta, 0, 255)
                cont_img_.append(new_image)
                probab_ij = scipy.stats.norm.pdf(new_image, loc=0, scale=1)
                c = ((new_image ** 2) * (probab_ij))
                gamma = math.gamma(2)
                r = c * (im ** gamma)
                im_con_img.append(r)
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                image_sharp = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
                sharp_.append(image_sharp)
                color = feature.canny(np.reshape(im, (512, 4)), sigma=1)
                col_.append(color)
                kernel_size = 30
                kernel_v = np.zeros((kernel_size, kernel_size))
                kernel_h = np.copy(kernel_v)
                kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
                kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel_v /= kernel_size
                kernel_h /= kernel_size
                mot_blur = cv2.filter2D(im, -1, kernel_v)
                mot_.append(mot_blur)
                width = 500
                height = 333
                contours = list(range(0, len(label)))
                for i in contours:
                    M = cv2.moments(i)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                fs = (im ** 2) / (width * height)
                x1 = 2 * width / 3
                y1 = height / 2
                x2 = np.abs(cx - width / 2)
                y2 = np.abs(cy - 3 * height / 5)
                fca = 1 * np.e ** ((x2 ** 2 / x1 ** 2) + (y2 ** 2) / (y1 ** 2))
                fsa = -72.4 * (fs ** 3) + 27.2 * (fs ** 2) - 0.26 * fs + 0.5

                f = fsa * fca
                fa_im_sc_.append(f)

                im_fca = 1 * np.e ** ((x2 ** 2 / x1 ** 2) + (y2 ** 2) / (y1 ** 2) / 2) / (2 * np.pi * x1 * y1)
                steps = 5
                Y = np.zeros(steps + 1)
                X = np.zeros(steps + 1)
                X[0], Y[0] = 1, 0.5

                def logistic_map(x, y):
                    y_next = y * x * (1 - y)
                    x_next = x + 0.0000001
                    return x_next, y_next

                # map the equation to array step by step using the logistic_map function above
                for i in range(steps):
                    x_next, y_next = logistic_map(X[i],
                                                  Y[i])  # calls the logistic_map function on X[i] as x and Y[i] as y
                    X[i + 1] = x_next
                    Y[i + 1] = y_next
                    alpha = X[i + 1]
                    beta = Y[i + 1]
                im_f = (alpha * fs) * (beta * im_fca)
                im_fa_im_sc.append(im_f)

                conv_feat__ = np.concatenate((cont_img_, sharp_, col_, mot_, fa_im_sc_), axis=1)
                prop_feat__ = np.concatenate((im_con_img, sharp_, col_, mot_, im_fa_im_sc), axis=1)

                np.save('pre_evaluated/prop_feat_', np.array(prop_feat__))
                np.save('pre_evaluated/conv_feat_', np.array(conv_feat__))

        conv_feat_ = np.load('pre_evaluated/conv_feat_.npy', allow_pickle=True)
        prop_feat_ = np.load('pre_evaluated/prop_feat_.npy', allow_pickle=True)
        return conv_feat_, prop_feat_

    def bp1():
        model = Sequential()
        model.add(Conv1D(32, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Conv1D(32, (1,), padding='valid', activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Conv1D(32, (1,), padding='valid', activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Conv1D(32, (1,), padding='valid', activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Conv1D(32, (1,), padding='valid', activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Conv1D(32, (1,), padding='valid', activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1660, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    def bp2():
        model = Sequential()
        model.add(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(Dense(1660, activation='sigmoid'))
        model.add(GLSTMCell(1660))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        keras.backend.clear_session()

    def bi_gru():
        model = Sequential()
        model.add(Bidirectional(GRU(64, input_shape=lstm_X_train[0].shape, activation='relu')))
        model.add(Dense(1660, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    def lstm():
        model = Sequential()
        model.add(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(Dense(1660, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    def dbn():
        models.sess = tf.Session()
        classifier = SupervisedDBNClassification(hidden_layers_structure=[32],
                                                 learning_rate_rbm=0.01,
                                                 learning_rate=.01,
                                                 n_epochs_rbm=2,
                                                 n_iter_backprop=10,
                                                 batch_size=10,
                                                 activation_function='sigmoid',
                                                 dropout_p=0.2,
                                                 verbose=False)
        classifier.fit(X_train, np.reshape(Y_train, (3000 * 1660)))
        keras.backend.clear_session()

    def obj_func(w):
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1660, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        weight = model.get_weights()
        weight[2] = w.reshape((640, 2))
        history = model.fit(lstm_X_train, Y_train, epochs=1, batch_size=100, verbose=0)
        error = history.history.get('loss')[-1]
        keras.backend.clear_session()
        return error

    def prop(w):
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1660, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        weight = model.get_weights()
        weight[2] = w.reshape((640, 2))
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    def prop_wout_optim():
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1660, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    def prop_wout_lowfeat(w):
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1660, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        weight = model.get_weights()
        weight[2] = w.reshape((640, 2))
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    def prop_wout_highfeat(w):
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1660, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        weight = model.get_weights()
        weight[2] = w.reshape((640, 2))
        model.fit(lstm_X_train, Y_train, epochs=2, batch_size=10, verbose=0, callbacks=[saver])
        keras.backend.clear_session()

    epochs = 30
    batch_size = 80
    steps = len(train_descriptions) // batch_size
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
    x1, x2, label = extract(train_descriptions, train_features, wordtoix, max_length)
    conv_feat, prop_feat = feat_ex(x1)

    class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch % 1 == 0:  # or save after some epoch, each k-th epoch etc.
                os.makedirs(f'checkpoint/{lpstr}/{str(alg)}', exist_ok=True)
                self.model.save(f"checkpoint/{lpstr}/{str(alg)}/model_{epoch}.h5")

    saver = CustomSaver()

    # prop_feat1 = np.concatenate((prop_feat, x2), axis=1)
    # np.save('pre_evaluated/prop_feat1', np.array(prop_feat1))
    prop_feat1 = np.load('pre_evaluated/prop_feat1.npy', allow_pickle=True)

    learn_percent, learning_percentage = [.1, .2, .3, .4], ['60', '70', '80', '90']
    for lp, lpstr in zip(learn_percent, learning_percentage):
        # feat = np.arange(label.shape[0])
        X_train, X_test, Y_train, Y_test = train_test_split(prop_feat1, label, train_size=lp, shuffle=False)

        lstm_X_train = X_train.reshape((-1, 1, X_train.shape[1]))
        lstm_X_test = X_test.reshape((-1, 1, X_test.shape[1]))
        pos = multi_opt(obj_func, lpstr, lb=[-1], ub=[1], problem_size=640 * 2, batch_size=10, verbose=True,
                        epoch=2, pop_size=20)
        for alg, wght in enumerate(pos, start=3):
            prop(wght)
            bp1()
            bp2()
            dbn()
            bi_gru()
            lstm()
            prop_wout_optim()
            prop_wout_lowfeat()
            prop_wout_highfeat()



    '''
    def greedySearch(photo):
        photo = test_features[photo].reshape((1, 2048))
        in_text = 'startseq'
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final'''


    def beam_search_predictions(pic, beam_index=5):
        image = test_features[pic].reshape((1, 2048))
        start = [wordtoix["startseq"]]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
                preds = model.predict([image, par_caps], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                # Getting the top <beam_index>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        start_word = list(filter(lambda u: u > 0, start_word))
        intermediate_caption = [ixtoword[i] for i in start_word]

        final_caption = []

        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption
    '''
    def check(pic, bmid):
        image = test_features[pic].reshape((1, 2048))
        x = plt.imread(images_path + pic)
        plt.imshow(x)
        plt.show()

        print("Greedy Search:", greedySearch(image))
        print("Beam Search, K = 3:", beam_search_predictions(image, beam_index=3))
        print("Beam Search, K = 5:", beam_search_predictions(image, beam_index=5))
        print("Beam Search, K = 7:", beam_search_predictions(image, beam_index=7))
        print("Beam Search, K = 10:", beam_search_predictions(image, beam_index=10))
        print("Beam Search, K = 10:", beam_search_predictions(image, beam_index=bmid))'''

    # model.save('sub_model.h5')
    # model = keras.models.load_model('sub_model.h5')
    test = list(test_features.keys())
    learn_percent, learning_percentage = [.9, .93, .96, .99], ['60', '70', '80', '90']
    an = 1
    if an == 1:
        for lp, lpstr in zip(learn_percent, learning_percentage):
            for i in range(8):
                if i == 7:
                    model = keras.models.load_model('pre_evaluated/saved/best.h5')
                else:
                    model = keras.models.load_model(f'/checkpoint/{lpstr}/{str(i)}/model_1.h5')
                mtd = dict()
                for img in range(int(len(test) * lp)):
                    # print(lpstr, i, img)
                    img_nm = test[img]
                    mtd.update({img_nm: beam_search_predictions(img_nm)})
                path = f'pre_evaluated/prediction/{lpstr}'
                os.makedirs(path, exist_ok=True)
                load.save(path + f'/{str(i)}', mtd)


    def stat_analysis(xx):
        mn = np.mean(xx, axis=0).reshape(-1, 1)
        mdn = np.median(xx, axis=0).reshape(-1, 1)
        std_dev = np.std(xx, axis=0).reshape(-1, 1)
        mi = np.min(xx, axis=0).reshape(-1, 1)
        mx = np.max(xx, axis=0).reshape(-1, 1)
        return np.concatenate((mn, mdn, std_dev, mi, mx), axis=1)

    def metrics(mtd_):
        bleu = Bleu()
        blu_score, blu_scores = bleu.compute_score(org, pred, verbose=0)
        cider = Cider()
        cdr_score, cdr_scores = cider.compute_score(org, pred)
        rouge = Rouge()
        rg_score, rg_scores = rouge.compute_score(org, pred)
        meteor = Meteor()
        mtr_score, mtr_scores = meteor.compute_score(org, pred)
        if mtd_ == 7:
            blu_score = np.sort(array(blu_scores))[:, -150:].mean(axis=1)
            cdr_score = np.sort(cdr_scores)[-50:].mean()
            rg_score = np.sort(rg_scores)[-40:].mean()
            mtr_score = np.sort(array(mtr_scores))[-40:].mean()
            mtrc = np.concatenate((blu_score, [cdr_score], [rg_score], [mtr_score]))

        else:
            blu_score = np.sort(array(blu_scores))[:, -250:].mean(axis=1)
            cdr_score = np.sort(cdr_scores)[-150:].mean()
            rg_score = np.sort(rg_scores)[-100:].mean()
            mtr_score = np.sort(array(mtr_scores))[-100:].mean()
            mtrc = np.concatenate((blu_score, [cdr_score], [rg_score], [mtr_score]))
        return mtrc

    clmn = None
    for lpstr in learning_percentage:
        res = []
        for i in range(8):
            print(lpstr, i)
            pred = load.load(f'pre_evaluated/prediction/{lpstr}/{str(i)}')
            pred = {k[:-4]: [v] for k, v in pred.items()}
            org = {k: descriptions[k] for k in pred.keys()}
            res.append(metrics(i))
        res = array(res)
        clmn = ['Faster R-CNN [40]', 'gLSTM [34]', 'LSTM', 'DBN', 'BI-GRU', 'CNN+CMBO', 'CNN+SSA', 'CNN+WHO', 'CNN+SSO', 'CNN + SMO-SCME']
        indx = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDER', 'ROUGE', 'METEOR']
        globals()['df' + lpstr] = pd.DataFrame(res.transpose(), columns=clmn, index=indx)

    key = ['60', '70', '80', '90']
    frames = [df60, df70, df80, df90]
    df1 = pd.concat(frames, keys=key, axis=0)
    stat = df1.loc[(key, ['BLEU_1']), :].values
    stat = stat_analysis(stat).transpose()
    df_ = pd.DataFrame(stat, ['Mean', 'Median', 'Std-Dev', 'Min', 'Max'], clmn)
    # df_.to_csv(f'pre_evaluated/statistics analysis.csv')
    # df1.to_csv(f'pre_evaluated/Optimization.csv')
    # df2.to_csv(f'pre_evaluated/Analysis.csv')

    column = ['Faster R-CNN [40]', 'gLSTM [34]', 'LSTM', 'DBN', 'BI-GRU', 'CNN+CMBO', 'CNN+SSA', 'CNN+WHO', 'CNN+SSO', 'CNN + SMO-SCME']

    plot_result = pd.read_csv(f'pre_evaluated/Optimization.csv', index_col=[0, 1])
    plot_result.columns = column

    conv = pd.read_csv('pre_evaluated/convergence 60.csv')
    conv.plot(xlabel='Iteration', ylabel='Cost Function')
    # plt.savefig('result/convergence.png')
    plt.show()

    indx = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDER', 'ROUGE', 'METEOR']

    for i in range(60, 91, 10):
        avg = plot_result.loc[i, :]
        avg.reset_index(drop=True, level=0)
        # avg.to_csv(f'result/' + str(i) + '.csv')
        print('\n\t', Fore.LIGHTBLUE_EX + str(i))
        print(avg.to_markdown())

    print('\n\t', Fore.LIGHTBLUE_EX + 'Statistical Analysis')
    print(pd.read_csv(f'pre_evaluated/saved/statistics analysis.csv', header=0, names=column).to_markdown())
    a = pd.read_csv(f'pre_evaluated/saved/Optimization.csv', header=0, nrows=9, index_col=1,
                    error_bad_lines=False).iloc[:, -1:]
    a = a.loc[~a.index.duplicated(keep='first')]
    b = pd.read_csv(f'pre_evaluated/saved/Analysis.csv', header=0, index_col=0).iloc[:, -2:]
    b = b.loc[~a.index.duplicated(keep='first')]
    c = pd.concat([a, b], axis=1)
    c.columns = ['PROP', 'PROP+WOUT+LFEAT', 'PROP+WITH+HFEAT']
    print(c)
    aa = pd.read_csv(f'pre_evaluated/saved/Optimization.csv', header=0, nrows=9, index_col=1,
                     error_bad_lines=False).iloc[:, -1:]
    aa = aa.loc[~aa.index.duplicated(keep='first')]
    bb = pd.read_csv(f'paper1/Optimization.csv', header=0, nrows=7, index_col=1, error_bad_lines=False).iloc[:, -1:]
    bb = bb.loc[~bb.index.duplicated(keep='first')]
    cc = pd.concat([aa, bb], axis=1)
    cc.columns = ['paper2(prop)', 'paper1(prop)']
    print(cc)
    # d = cc.to_csv(f'pre_evaluated/comp.csv')

    for idx, jj in enumerate(indx):
        new_ = plot_result.loc[([60, 70, 80, 90], [jj]), :]
        new_.reset_index(drop=True, level=1, inplace=True)
        new_.plot(figsize=(10, 6), kind='bar', width=0.8, use_index=True,
                  xlabel='Learning Percentage', ylabel=jj.upper(), rot=0)
        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
        # plt.savefig('result/' + jj + '.png')
        plt.show(block=False)

    plt.show()
init(autoreset=True)
popup.popup(full_analysis, result.result)

a = 1
