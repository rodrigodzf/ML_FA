import tensorflow as tf;
import numpy as np

def main():
    core_tutorial()

# https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial
def core_tutorial():
    sess = tf.Session()

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    add_and_triple = adder_node * 3.

    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    init = tf.global_variables_initializer()
    sess.run(init)

    # placeholder to provide the desired values
    y = tf.placeholder(tf.float32)

    # loss function measures how far apart the current model is from the provided data.
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    # fixW = tf.assign(W, [-1.])
    # fixb = tf.assign(b, [1.])
    # sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    ## Training
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess.run(init)  # reset values to incorrect defaults.
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([W, b]))



if __name__ == '__main__':
    main()