from __future__ import print_function
import tensorflow as tf


def tf_hello_world():
    hello = tf.constant('Hello, TensorFlow!')

    # Start tf session
    sess = tf.Session()

    # Run the op
    print(sess.run(hello))


def tf_hello_world2():
    hello = tf.constant('Hello, TensorFlow2!')

    # Start tf session
    with tf.compat.v1.Session() as sess:

        # Run the op
        print(sess.run(hello))