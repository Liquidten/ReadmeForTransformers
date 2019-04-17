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