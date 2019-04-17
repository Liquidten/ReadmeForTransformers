# Preprocessing Data

    #importing imdb data from keras datasert 
    imdb = keras.datasets.imdb
    
    #num_words = 10000 keeps the 10,000 most frequent occured words from the training data 
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    return (train_data, train_labels), (test_data, test_labels)
    
    #Helper function to convert integers back to words
    #A dictonary mapping words to an integer index
    
    word_index = imdb.get_word_index()

    #The first indices are reserved
    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2 #unknown
    word_index["<UNUSED>"] = 3

    text_data = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([text_data.get(i, '?') for i in text])
    
    # Data Padding to same length sequences
    data_padding(train_data,test_data):
    #Data Preperation, padding the array so all of the data has the same length
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding = 'post',
                                                            maxlen = 300)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding = 'post',
                                                           maxlen = 300)
    
    


# Model Design
### CNN Model: 

    #Building a 1D Convvolutional Neural Network Model
    #input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000
    maxlen = 300
    embedding_vector_length = 32
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_vector_length,
                                     input_length=maxlen))
    model.add(keras.layers.Conv1D(32,kernel_size=(3),strides=1, padding='same',
                                  activation= 'relu'))
    model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.Conv1D(64, kernel_size=(3),strides=1, padding='same', activation='relu'))
    model.add(keras.layers.AveragePooling1D(pool_size = (2)))
    #Randomely dropping neurons to improve convergenc
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
    model.summary()
    
    Adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = Adam,loss='binary_crossentropy', metrics=['acc', auc])

    #splitting the data for validation purposes 
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    pratial_y_train = train_labels[10000:]

    #Training the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                                                   verbose=0, mode='auto', baseline=None, 
                                                   restore_best_weights=False)
    history = model.fit(np.array(partial_x_train),np.array(pratial_y_train),epochs=40, batch_size=512, 
                        validation_data=(np.array(x_val),np.array(y_val)),
                        verbose=1, callbacks=[early_stopping])

### RNN Model: 

    # Input the value, whether you want to run the model on LSTM RNN or GRU RNN.
    print("Input 'LSTM' for LSTM RNN, 'GRU' for GRU RNN ")
    modelInput= input("Do you want to compile the model using LSTM RNN or GRU RNN?\n")
    if modelInput == "LSTM":
        lstm = True
    else:
        lstm = False
	#Building a 1D Convvolutional Neural Network Model
    #input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000
    maxlen = 300
    embedding_vector_length = 32
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_vector_length, input_length=maxlen))
    #model.add(keras.layers.Dropout(0.2))
    #LSTMmodel.add(keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    #model.add(keras.layers.MaxPool1D(pool_size = 2))
    if lstm == True:
        model.add(keras.layers.LSTM(150))
    else:
        model.add(keras.layers.GRU(150))    
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

### Transformer Model: 

	t2t-trainer \
	--data_dir=data_dir \
	--output_dir=train_dir \
	--problem=sentiment_imdb \
	--model=transformer_encoder \
	--hparams_set=transformer_base \
	--train_steps=1000 \
	--eval_steps=100

	# generate data
	t2t-datagen \
	--data_dir=data_dir \
	--tmp_dir=tmp_dir \
	--problem=translate_enzh_wmt32k
	
	# use the trainer
	t2t-trainer \
	--data_dir=data_dir \
	--output_dir=train_dir \
	--problem=translate_enzh_wmt32k \
	--model=transformer_encoder \
	--hparams_set=transformer_base \
	--train_steps=1000 \
	--eval_steps=100

	# build decode
	DECODE_FILE=data_dir/decode_this.txt
	echo “I love algorithm” >> $DECODE_FILE
	echo “Dr.Albert is good professor.” >> $DECODE_FILE

	# decode it
	t2t-decoder \
	--data_dir=data_dir \
	--problem=translate_enzh_wmt32k \
	--model=transformer_encoder \
	--hparams_set=transformer_base \
	--output_dir=train_dir \
	--decode_hparams=“beam_size=4,alpha=0.6” \
	--decode_from_file=$DECODE_FILE \
	 --decode_to_file=translation.en

	# See the translations
	cat translation.en
