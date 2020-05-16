import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only output the error information

plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    # Load the MNIST dataset
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # print(tf.reduce_min(x), tf.reduce_max(x))
    # print(tf.reduce_min(y), tf.reduce_max(y))

    # Convert to a floating point tensor and scale to 0~1
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.0

    # Convert to an integer tensor
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    # one-hot coding
    y = tf.one_hot(y, depth=10)

    # [b, 28, 28] => [b, 28*28]
    x = tf.reshape(x, (-1, 28 * 28))

    # Build the dataset object
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(200)

    return train_dataset

def init_paramaters():
    '''
    The tensors of each layer need to be optimized,
    so the Variable type is used and the truncated normal distribution is used to initialize the weight tensor
    '''
    # The parameters of the first layer
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))

    # The parameters of the second layer
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))

    # The parameters of the third layer
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    return w1, b1, w2, b2, w3, b3

def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # First layer
            # x: [b, 28*28]
            # [b, 784] @ [784, 256] + [256] -> [b, 256] + [256] = [b, 256] + [b, 256]
            h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)

            # [b, 256] -> [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            # [b, 256] -> [b, 128]
            # out = ([10, 200])
            out = h2 @ w3 + b3

            # compute loss
            # mse = mean(sum(y-out)^2)
            loss = tf.square(y - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)

            # The tensors of the required gradient are [w1, b1, w2, b2, w3, b3]

            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # updates in place (w1 = w1 - lr * w1_grads)
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss is: ', loss.numpy())

    return loss.numpy()

def train(lr, epochs):
    losses = []
    train_dataset = load_data()
    w1, b1, w2, b2, w3, b3 = init_paramaters()

    for epoch in range(epochs):
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr)
        losses.append(loss)

    x = [i for i in range(0, epochs)]
    # plot figure
    plt.plot(x, losses, color='blue', marker='s', label='train')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MNIST_loss.png')
    plt.show()
    plt.close()

if __name__ == '__main__':

    train(0.001, 20)