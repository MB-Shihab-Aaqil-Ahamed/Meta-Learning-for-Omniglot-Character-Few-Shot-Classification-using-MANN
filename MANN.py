import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator

from tensorflow.keras import layers

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import datetime

def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####

    # Only on last N example, corresponding to the K+1 sample
    # print("> Predds:", preds[:, -1:, :, :])
    # print("> Labels:", labels[:, -1:, :, :])
    loss = tf.keras.losses.categorical_crossentropy(y_true=labels[:, -1:, :, :],
                                                    y_pred=preds[:, -1:, :, :],
                                                    from_logits=True)
    loss = tf.reduce_sum(loss)
    # print(loss)
    return loss
    #############################

class MANN(tf.keras.Model):

    def __init__(self, num_classes, num_samples_per_class):

        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            inputs_images: [B, K+1, N, 784] flattened images
            input_labels: [B, K+1, N, N] ground truth labels
        Returns:
            preds: [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        # print(input_labels)
        # print(">>> input_images.shape", input_images.shape)

        _, K, N, I = input_images.shape

        # Zero last N example, corresponding to the K+1 sample
        # Note the -1: so num of dimensions keeps equal
        in_zero = input_labels - input_labels[:, -1:, :, :]
        # tf.print(in_zero)

        input = tf.keras.layers.Concatenate(axis=3)([input_images, in_zero])
        # print(input.shape)

        input = tf.reshape(input, [-1, K*N, N + 28*28])
        # print(input.shape)

        out = self.layer2(self.layer1(input))
        out = tf.reshape(out, [-1, K, N, N])
        # print('> out:', out)
        #############################
        return out

def run_experiment(num_classes=5, num_samples_per_class=1, meta_batch_size=16, epochs=5e5, verbose=True):
    #Placeholders for images and labels

    ims = tf.compat.v1.placeholder(tf.float32, shape=(None, num_samples_per_class + 1, num_classes, 784))
    labels = tf.compat.v1.placeholder(tf.float32, shape=(None, num_samples_per_class + 1, num_classes, num_classes))

    data_generator = DataGenerator(num_classes, num_samples_per_class + 1)

    o = MANN(num_classes, num_samples_per_class + 1)
    out = o(ims, labels)

    loss = loss_function(out, labels)

    optim = tf.train.AdamOptimizer(0.001)
    optimizer_step = optim.minimize(loss)
    # print("... Starts training ...")
    last_time = datetime.datetime.now()
    print("Time:", last_time.time())

    # For plotting
    steps = []
    train_losses = []
    test_losses = []
    test_accurs = []

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Train and test:
        for step in range(int(epochs)):
            i, l = data_generator.sample_batch('train', meta_batch_size)
            # print("i.shape:",i.shape)

            feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
            # print("feed[ims].shape:", feed[ims].shape)
            _, ls = sess.run([optimizer_step, loss], feed)

            if step % 100 == 0 or step == int(epochs)-1:
                # print("*" * 5 + "Iter " + str(step) + "*" * 5)
                i, l = data_generator.sample_batch('test', 1000)
                feed = {ims: i.astype(np.float32),
                        labels: l.astype(np.float32)}
                pred, tls = sess.run([out, loss], feed)
                # print("Train Loss:", ls, "Test Loss:", tls)
                pred = pred.reshape(
                    -1, num_samples_per_class + 1,
                    num_classes, num_classes)
                pred = pred[:, -1, :, :].argmax(2)
                l = l[:, -1, :, :].argmax(2)
                # print("Test Accuracy", (1.0 * (pred == l)).mean())
                # For plotting
                steps.append(step)
                train_losses.append(ls)
                test_losses.append(tls)
                test_accurs.append((1.0 * (pred == l)).mean())

    
    # plt.plot(steps, train_losses)
    # plt.plot(steps, test_losses)
    # plt.show()

    print("Time:", last_time.time())
    print("... Training ended ...")

    return steps, train_losses, test_losses, test_accurs

# Number of classes used in classification (e.g. 5-way classification).
num_classes = 5

# Number of examples used for inner gradient update (K for K-shot learning)
num_samples_per_class = 1

# Number of N-way classification tasks per batch
meta_batch_size = 16

steps, train_losses, test_losses, test_accurs = run_experiment(epochs=1e3)

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Training Loss", "Test Loss", "Test Accuracy", ""))

fig.add_trace(go.Scatter(x=steps, y=train_losses, name='Train Err'),
            row=1, col=1
            )
fig.add_trace(go.Scatter(x=steps, y=test_losses, name='Test Err'),
            row=1, col=2
            )
fig.add_trace(go.Scatter(x=steps, y=test_accurs, name='Test Acc'),
            row=2, col=1
            )

fig.update_layout(title_text="Performance over time")
fig.show()
    

    