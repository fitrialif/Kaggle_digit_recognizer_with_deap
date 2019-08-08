# import statements needed for the functions below
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import deap_functions # this is done to avoid circular import error
from lenet5 import le_net_5
import time
from sklearn.utils import shuffle


# this function is used for fast evaluation of the individuals
# during population creation and maybe later on when checking if
# child or mutant is worth full evaluation
# it is fast because it limits evaluation to 10s
# individual is an object with all the genes
def evaluate_individual_fast(individual, train_dataset, valid_dataset, test_dataset,
                             train_labels, valid_labels, test_labels):
    max_time = 10
    accuracy = evaluate_individual(individual, max_time, train_dataset, valid_dataset, test_dataset,
                                   train_labels, valid_labels, test_labels)
    return accuracy


# decorator for the function evaluate_individual
# it is doing normal (slow) evaluation of 100s
# individual is an object with all the genes
def evaluate_individual_normal(individual, train_dataset, valid_dataset, test_dataset,
                               train_labels, valid_labels, test_labels):
    max_time = 40
    accuracy = evaluate_individual(individual, max_time, train_dataset, valid_dataset, test_dataset,
                                   train_labels, valid_labels, test_labels)
    return accuracy


# this function is used exclusively in evaluate_individual
# can be used to evaluate accuracy of every iteration and
# is used for evaluation at the end and returned value is used as fitness of the individual
# it performs evaluation of the batch
# x and y_ are placeholders defined for tensorflow
# x_data -
# y_data
# batch_size
# accuracy_operation  - mean value calculated as float - not sure why this is needed
def evaluate(x_data, y_data, batch_size, accuracy_operation, x, y_):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# this is main evaluation function
# it is decorated by two small functions above
def evaluate_individual(individual, max_time, train_dataset, valid_dataset, test_dataset,
                        train_labels, valid_labels, test_labels):

    #print(individual)
    tf.reset_default_graph()

    # extract genes from the individual
    batch_size, learning_rate, standard_dev, receptive_field_1, filters_1, receptive_field_2, \
        filters_2, activation = [x for x in individual]

    # here calculate padding depending on the size of the receptive field
    padding_size = int((receptive_field_1 - 1) / 2)
    # print(padding_size)

    # Pad images with 0s
    from data_load import padding
    train_dataset_padded = padding(train_dataset, padding_size)
    valid_dataset_padded = padding(valid_dataset, padding_size)
    test_dataset_padded = padding(test_dataset, padding_size)
    # submit_dataset_padded = padding(submit_dataset, padding_size)

    x = tf.placeholder(tf.float32, shape=[None, (28 + padding_size * 2), (28 + padding_size * 2), 1])
    y_ = tf.placeholder(tf.int32, None)

    # Invoke LeNet function by passing features
    logits = le_net_5(x, standard_dev, receptive_field_1, filters_1, receptive_field_2, filters_2, activation)

    # Softmax with cost function implementation
    # I think it does calculate error - measures difference between labels and output of the network
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)

    loss_operation = tf.reduce_mean(cross_entropy)
    # Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
    # to update network weights iterative based in training data.
    # it is about changing weights to get better accuracy
    # optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # here I pass to the Adam calculated error (cross_entropy)
    training_operation = optimizer.minimize(loss_operation)

    # here i compare results between between results from the network and labels
    # the tf.argmax is needed because results from network are probably not binary, I guess that
    # they come as probability and this tf.argmax just gets highest probability
    # not sure why we need argmax for labels... mystery...
    # it returns boolean tensor, something like 1 0 0 0 1 1 1 1 0 0 0 etc
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

    # here mean is calculated but for some unclear reason it needs input of float
    # maybe if it was not casted to float it would return mean as integer??
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    with tf.Session() as sess:
        #print("nodes of the graph: ", len(sess.graph._nodes_by_name.keys()))
        # tf.global_variables_initializer Returns an Op that initializes global variables.
        sess.run(tf.global_variables_initializer())
        num_examples = len(train_dataset_padded)

        # i guess this does not work, i see that almost all GPU ram is allocated
        # while it should allocate only minimum needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # print("Training... with dataset - ", num_examples)
        # print()
        # start_total = time.time()
        time_used = 0
        # start_time = time.time()
        start = time.time()
        i = 0
        # this loop is supposed to last until we do not run out of time
        while time_used < max_time:
            # for i in range(EPOCHS):

            train_dataset_padded_shuffled, train_labels_shuffled = shuffle(train_dataset_padded, train_labels)

            # here  I believe new batches is being prepared and send for calculation
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = train_dataset_padded_shuffled[offset:end], train_labels_shuffled[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y_: batch_y})

            # stop = time.time()
            # print("training {:.5f} ...".format(stop-start))

            # start = time.time()
            validation_accuracy = evaluate(valid_dataset_padded, valid_labels, batch_size, accuracy_operation, x, y_)
            # stop = time.time()
            # print("evaluating  {:.5f} ...".format(stop-start))
            # print("EPOCH {} ...".format(i+1))
            #print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            # print()
            time_used = time.time() - start
            #print("time_used {:.5f}" .format(time_used))
            i += 1
        # stop_total = time.time()
        # print("Total time {:.5f}".format(stop_total-start_total))

        # saver = tf.train.Saver()
        # save_path = saver.save(sess, '/tmp/lenet.ckpt')
        # print("Model saved %s "%save_path)

        test_accuracy = evaluate(test_dataset_padded, test_labels, batch_size, accuracy_operation, x, y_)
        #print("Test Accuracy = {:.3f}".format(test_accuracy))

    return test_accuracy,  # do not remove coma, deap expect tuple rather than float
