from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.optimizers import *

def get_model(prprocessingHelper):
    VOCABULARY_SIZE = len(preprocessingHelper.le.classes_) + 1
    output_dim = 2
    
    input = Input(shape=(preprocessingHelper.max_sequence_length, len(preprocessingHelper.feature_cols)))
    M1 = Masking(mask_value=0.0, input_shape=(preprocessingHelper.max_sequence_length, len(preprocessingHelper.feature_cols)))(input)
    
    E1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=output_dim, trainable=True, input_length=preprocessingHelper.max_sequence_length)(M1[:, :, 0])
    E2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=output_dim, trainable=True, input_length=preprocessingHelper.max_sequence_length)(M1[:, :, 1])
    
    C1 = Concatenate()([E1, E2, M1[:, :, 2:]])
    L1 = LSTM(64, return_sequences=True)(C1)
    output = Dense(1, activation='sigmoid')(L1)
    lstm_model = Model(inputs=input, outputs=output)

