import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

assert tf.__version__.startswith('2.')       # Tensorflow 2

def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x,y

def build_model(batch_size):
    # Get training data
    (x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
    print(x.shape, y.shape)
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(10000).batch(batch_size)
    # For test data
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(batch_size)   # Don't need to shuffle
    # For data visualization
    db_iter = iter(db)
    sample = next(db_iter)
    print('batch:', sample[0].shape, sample[1].shape)

    # Build dense layer
    model = Sequential([
        layers.Dense(256, activation = tf.nn.relu), # [b, 784] => [b, 256]
        layers.Dense(128, activation = tf.nn.relu), # [b, 256] => [b, 128]
        layers.Dense(64, activation = tf.nn.relu),  # [b, 128] => [b, 64]
        layers.Dense(32, activation = tf.nn.relu),  # [b, 64] => [b, 32]
        layers.Dense(10)                          # [b, 32] => [b, 10], 330 = 32*10 + 10
    ])
    model.build(input_shape=[None, 28 * 28])
    model.summary()
    # w = w - lr * grad
    optimizer = optimizers.Adam(lr = 1e-3)   # Stochastic gradient descent

    # Training
    for epoch in range(40):
        for step, (x, y) in enumerate(db):
            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)                   # Forward Propagation
                y_onehot = tf.one_hot(y, depth = 10)
                # Loss functions
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            # Update gradient in place
            optimizer.apply_gradients(zip(grads, model.trainable_variables))        # Processing aggregated gradients

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))

        # For test

        total_correct = 0
        total_num = 0

        for step, (x_test, y_test) in enumerate(db_test):

            # x: [b, 28, 28] => [b, 784]
            # y: [b]

            x_test = tf.reshape(x_test, [-1, 28*28])
            logits = model(x)      # Forward Propagation (Get the output value)

            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)

            # [b, 10] => [b], int64                        # (The position number is the number to be predicted)
            pred = tf.argmax(prob, axis=1)                 # Returns the index with the largest value across axes of a tensor.
            pred = tf.cast(pred, dtype=tf.int32)

            # pred:[b]    using softmax function
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)                    # The highest probability of that position is this number
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)


if __name__ == '__main__':

    build_model(128)