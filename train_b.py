import numpy as np
import tensorflow as tf

from train_a import load_data, output_results, build_model


CHOICE_P_SIZE = 40000
EPOCH = 10
BATCH_SIZE = 200
INNER_LAYER_ACTIVATION = 'relu'


if __name__ == '__main__':
    
    train_x , train_y, train_x_normalized, train_y_normalized = load_data(
        path_image='data/train-images-idx3-ubyte',
        path_label='data/train-labels-idx1-ubyte',
        size=CHOICE_P_SIZE,
    )

    model = build_model(INNER_LAYER_ACTIVATION, output_shape=1, loss="mse")
    
    model_for_normalized = build_model(INNER_LAYER_ACTIVATION)
    
    history = model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE)
    
    history_2 = model_for_normalized.fit(train_x_normalized, train_y_normalized, epochs=EPOCH, batch_size=BATCH_SIZE)
    
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    accuracy_2 = history_2.history['accuracy']
    loss_2 = history_2.history['loss']
    
    output_results(datasets=[loss, loss_2], title='B_cost_history', y_label='cost', legend=['standard', 'normalized'])
    output_results(datasets=[accuracy, accuracy_2], title='B_accuracy_history', y_label='accuracy', legend=['standard', 'normalized'])
    
    
    
    