import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from mlxtend.data import loadlocal_mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical



CHOICE_P_SIZE = 40000
EPOCH = 10
BATCH_SIZE = 1
INNER_LAYER_ACTIVATION = 'sigmoid'



def load_data(path_image, path_label, size):
    x , y = loadlocal_mnist(
        images_path=path_image,
        labels_path=path_label
    )
    idx = np.random.choice(x.shape[0], size, replace=False)
    choice_x = x[idx]
    choice_y = y[idx]
    normalized_x = choice_x / 255.0
    return choice_x.reshape(-1, 28, 28, 1), choice_y, normalized_x.reshape(-1, 28, 28, 1), to_categorical(choice_y)


def output_results(datasets, title='_', x_label='Epoch', y_label='Y', legend=[]):
    _xlimt = 10
    for _ in datasets:
        _len = len(_)
        _xlimt = max(_len, _xlimt)
        plt.plot(range(_len), _)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim([0, _xlimt])
    
    if legend:
        plt.legend(legend, loc='upper right')
    if title:
        plt.title(title)
        
    plt.savefig('output/{}.jpg'.format(title))
    plt.show()
    plt.clf()
    


def build_model(inner_activation, output_shape = 10, loss = 'categorical_crossentropy'):
    model = Sequential([
        Conv2D(10, (3,3), padding='same', input_shape=(28, 28, 1)),
        # MaxPooling2D(2, 2),
        Dense(10, activation=inner_activation),
        Dense(10, activation=inner_activation),
        # Dropout(0.2),
        # Dense(10, activation='softmax')
        Flatten(), Dense(output_shape, activation=inner_activation)
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
    
    # print(train_x.shape)
    # print(train_y.shape)

    model = build_model(INNER_LAYER_ACTIVATION, output_shape=1, loss="mse")
    
    model_for_normalized = build_model(INNER_LAYER_ACTIVATION, output_shape=10)
    
    history = model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE)
    
    history_2 = model_for_normalized.fit(train_x_normalized, train_y_normalized, epochs=EPOCH, batch_size=BATCH_SIZE)
    
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    accuracy_2 = history_2.history['accuracy']
    loss_2 = history_2.history['loss']
    
    print('accuracy: ', accuracy)
    print('loss: ', loss)
    
    output_results(datasets=[loss, loss_2], title='A_cost_history', y_label='cost', legend=['standard', 'normalized'])
    output_results(datasets=[accuracy, accuracy_2], title='A_accuracy_history', y_label='accuracy', legend=['standard', 'normalized'])
    
    
    
    