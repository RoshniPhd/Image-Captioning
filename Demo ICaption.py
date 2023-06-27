
# file_path = filedialog.askopenfilename()
# import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Merge, LSTM, Embedding,BatchNormalization, Dropout, TimeDistributed, Dense, RepeatVector, Activation, Flatten
from keras.layers.wrappers import Bidirectional
import numpy as np
import pandas as pd
import pickle
from PIL import Image


def bilstm_predictions(image_file, beam_index=3):
    start = [word_idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            now_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            try:
                e = encoding_test[image_file]
            except:
                e = encoding_train[image_file]
            preds = fin_model.predict([np.array([e]), np.array(now_caps)])

            word_preds = np.argsort(preds[0])[-beam_index:]

            # Getting the top Beam index = 3  predictions and creating a
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
    intermediate_caption = [idx_word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


# code to make caption dictionary whose keys are image file name and values are image caption.
token_dir = "Flickr8k_text/Flickr8k.token.txt"


encoding_train = pickle.load(open('encoded_train_images_inceptionV3.p', 'rb'))
encoding_test = pickle.load(open('encoded_test_images_inceptionV3.p', 'rb'))




dataframe = pd.read_csv('Flickr8k_text/trainimgs.txt', delimiter='\t') # read the trainimgs as dataframe
captionz = [] # tot_captions
img_id = [] # tot_image_ID
dataframe = dataframe.sample(frac=1)
iter = dataframe.iterrows() # dataframe have  image name id, caption

for i in range(len(dataframe)):
    nextiter = next(iter)
    captionz.append(nextiter[1][1]) # train image captions
    img_id.append(nextiter[1][0]) # train image id

no_samples=0
tokens = []   # split the captions into token by token
tokens = [i.split() for i in captionz]
for caption in captionz:
    no_samples+=len(caption.split())-1

vocab= []
#for token in tokens:
#    vocab.extend(token)
vocab = list(set(vocab))
#with open("vocab.p", "wb") as pickle_d:
#   pickle.dump(vocab, pickle_d)
vocab= pickle.load(open('vocab.p', 'rb'))
print (len(vocab))

vocab_size = len(vocab)
word_idx = {val:index for index, val in enumerate(vocab)} # each word with index
idx_word = {index:val for index, val in enumerate(vocab)}

caption_length = [len(caption.split()) for caption in captionz]
max_length = max(caption_length)


def Bidirectional_lstm (tr,vocab_size, DD2):
    EMBEDDING_DIM = 300
    # Model

    image_model = Sequential()
    image_model.add(Dense(EMBEDDING_DIM, input_shape=(2048,), activation='relu'))
    image_model.add(RepeatVector(max_length))

    lang_model = Sequential()
    lang_model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    lang_model.add(Bidirectional(LSTM(256, return_sequences=True)))
    lang_model.add(Dropout(0.5))
    lang_model.add(BatchNormalization())
    lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

    fin_model = Sequential()
    fin_model.add(Merge([image_model, lang_model], mode='concat'))
    fin_model.add(Dropout(0.5))
    fin_model.add(BatchNormalization())
    fin_model.add(Bidirectional(LSTM(1000, return_sequences=False)))

    fin_model.add(Dense(vocab_size))
    fin_model.add(Activation('softmax'))
    print("Model created!")
    fin_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # fin_model.load_weights("model_weights10.h5") # best is model10
    epoch = DD2
    batch_size = 128
    fin_model.load_weights("bidirectional.h5")
    return fin_model
tr=0
DD1 = np
fin_model = Bidirectional_lstm (tr,vocab_size, DD1)


# #---- caption for train_image ----#
# a=['2875528143_94d9480fdd.jpg',	'3556792157_d09d42bef7.jpg','3055716848_b253324afc.jpg','3272541970_ac0f1de274.jpg','2646615552_3aeeb2473b.jpg','3204712107_5a06a81002.jpg']
# for i in range(0,a.__len__()):
#     print(i)
#     # try_image = file_path
#     t = 'D:/Bommy/BOMMY PYTHON WORKS/Roshini Padate (75717) - Paper 3 (Class I)/PAPER3/Roshini_paper3_final code/Flickr8K_Data/'+ a[i]
#     im=Image.open(t)
#     im.show()
#     im.save('result/sso/sample' +str(i)+'.jpg')
#     l, m = t.rsplit('/', 1)
#     c=bilstm_predictions(m, beam_index=3)

    # file1 = 'D:/Bommy/BOMMY PYTHON WORKS/Roshini Padate (75717) - Paper 3 (Class I)/PAPER3/Roshini_paper3_final code/result/sso/sample' + str(i) + '.txt'
    #
    # f = open(file1, "w")
    #
    # f.write(c)
    # f.write("\n")
    # f.close()


