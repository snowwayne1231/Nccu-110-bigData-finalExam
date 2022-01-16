import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from train_a import load_data, output_results


CHOICE_P_SIZE = 40000
EPOCH = 100
BATCH_SIZE = 500
INNER_LAYER_ACTIVATION = 'relu'



def build_model(inner_activation, output_shape = 10, loss = 'categorical_crossentropy'):
    model = Sequential([
        Conv2D(32, (2,2), padding='same', input_shape=(28, 28, 1), activation=inner_activation),
        MaxPooling2D(2, 2),
        Conv2D(64, (2,2), padding='same', activation=inner_activation),
        Dense(100, activation=inner_activation),
        Dropout(0.2),
        Dense(100, activation=inner_activation),
        Flatten(), Dense(output_shape, activation='softmax')
    ])
    
    optimizer = RMSprop(
        learning_rate=0.002,
        rho=0.9,
        momentum=0.1,
        epsilon=1e-05,
        centered=True,
    )

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model



if __name__ == '__main__':
    
    train_x , train_y, train_x_normalized, train_y_normalized = load_data(
        path_image='data/train-images-idx3-ubyte',
        path_label='data/train-labels-idx1-ubyte',
        size=CHOICE_P_SIZE,
    )
    
    val_x , val_y, val_x_normalized, val_y_normalized = load_data(
        path_image='data/t10k-images-idx3-ubyte',
        path_label='data/t10k-labels-idx1-ubyte',
        size=10000,
    )                         

    model = build_model(INNER_LAYER_ACTIVATION)
    
    history = model.fit(
        train_x_normalized,
        train_y_normalized,
        epochs=EPOCH,
        validation_data=(
            val_x_normalized,
            val_y_normalized,
        ),
        batch_size=BATCH_SIZE,
    )
    
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    accuracy_2 = history.history['val_accuracy']
    loss_2 = history.history['val_loss']
    
    output_results(datasets=[loss, loss_2], title='C_cost_history', y_label='cost', legend=['basic', 'validation'])
    output_results(datasets=[accuracy, accuracy_2], title='C_accuracy_history', y_label='accuracy', legend=['basic', 'validation'])
    
    
    
    