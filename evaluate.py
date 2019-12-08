# import statements needed for the functions below
import os
import tensorflow as tf
from lenet5 import le_net_5
import time
from sklearn.utils import shuffle
from tensorflow.python.framework import ops
from data_load import padding

# i forgotten what was this about
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import deap_functions # this is done to avoid circular import error


# this function is used for fast evaluation of the individuals
# during population creation and maybe later on when checking if
# child or mutant is worth full evaluation
# it is fast because it limits evaluation to 10s
# individual is an object with all the genes
def train_individual_fast(individual, train_dataset, valid_dataset, test_dataset,
                          train_labels, valid_labels, test_labels):
    max_time = 10
    accuracy, protobuf, metagraph, x, y_, save_path = train_individual(individual, max_time, train_dataset, valid_dataset, test_dataset,
                                train_labels, valid_labels, test_labels)
    return accuracy, protobuf, metagraph, x, y_, save_path


# decorator for the function evaluate_individual
# it is doing normal (slow) evaluation of 100s
# individual is an object with all the genes
def train_individual_normal(individual, train_dataset, valid_dataset, test_dataset,
                            train_labels, valid_labels, test_labels):
    max_time = 40
    accuracy, protobuf, metagraph, x, y_, save_path = train_individual(individual, max_time, train_dataset, valid_dataset, test_dataset,
                                train_labels, valid_labels, test_labels)
    return accuracy, protobuf, metagraph, x, y_, save_path


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
    sess = tf.compat.v1.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    #print("am in evaluate")
    return total_accuracy / num_examples


# this is main evaluation function
# it is decorated by two small functions above
def train_individual(individual, max_time, train_dataset, valid_dataset, test_dataset,
                     train_labels, valid_labels, test_labels):

    # reset default graph otherwise system will keep leaking memory
    ops.reset_default_graph()
    # tf.reset_default_graph()

    # extract genes from the individual
    batch_size, learning_rate, standard_dev, receptive_field_1, filters_1, receptive_field_2, \
        filters_2, activation = [x for x in individual]

    # here calculate padding depending on the size of the receptive field
    padding_size = int((receptive_field_1 - 1) / 2)

    # Pad images with 0s
    train_dataset_padded = padding(train_dataset, padding_size)
    valid_dataset_padded = padding(valid_dataset, padding_size)
    test_dataset_padded = padding(test_dataset, padding_size)
    # submit_dataset_padded = padding(submit_dataset, padding_size)


    # x is placeholder for feedinig in images
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, (28 + padding_size * 2), (28 + padding_size * 2), 1])
    # y_ is a placeholder for ???
    y_ = tf.compat.v1.placeholder(tf.int32, None)


    # with tf.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     a = tf.print(x)
    #     #print(sess.run(x))
    #     #print(sess.run("x....", x))
    # print(a)

    #sess = tf.InteractiveSession()
    #with tf.InteractiveSession as sess:
    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     a = tf.Print(x, [x], "this is x")
    #     b = tf.multiply(a, 1)
    #     a.eval()

    # with tf.Session() as sess:  print("y....", y_)
    # print("x+++", x)
    # print("y_+++", y_)

    # with tf.Session() as sess:
    #     sess.run(init_op)  # execute init_op
    #     # print the random values that we sample
    #     print(sess.run(normal_rv))




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
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

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

    # allow process to consume only what it needs
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # define saver
    saver = tf.compat.v1.train.Saver()

    # this was my attempt to move to tensorflow2 but using old style from tensorflow1
    # with tf.compat.v1.Session() as sess:
    with tf.compat.v1.Session(config=config) as sess:
        # print("nodes of the graph: ", len(sess.graph._nodes_by_name.keys()))
        # tf.global_variables_initializer Returns an Op that initializes global variables.
        sess.run(tf.compat.v1.global_variables_initializer())
        num_examples = len(train_dataset_padded)

        # print("Training... with dataset - ", num_examples)
        # print()
        # start_total = time.time()
        time_used = 0
        # start_time = time.time()
        start = time.time()
        i = 0
        # this loop is supposed to last until we do not run out of time
        print("Validation Accuracy : ", end='')
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
            # validation set is 80 of the test set, no sur
            validation_accuracy = evaluate(valid_dataset_padded, valid_labels, batch_size, accuracy_operation, x, y_)
            # stop = time.time()
            # print("evaluating  {:.5f} ...".format(stop-start))
            # print("EPOCH {} ...".format(i+1))
            print(" {:.3f} ".format(validation_accuracy), end='')
            # print()
            time_used = time.time() - start
            # print("time_used {:.5f}" .format(time_used))
            i += 1

        stop_total = time.time()
        # print("Total time {:.5f}".format(stop_total-start_total))
        print(" Total time {:.5f}".format(stop_total-start))

        # move saver before starting session
        #saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, '/tmp/saver.ckpt')
        #save_path = saver.save(sess, savefile)
        print("Model saved %s "%save_path)
        protobuf = saver.to_proto(export_scope=None)
        #metagraph = saver.export_meta_graph()
        metagraph = saver.export_meta_graph(filename='/tmp/metagraph.txt', as_text=True)
        # test_accuracy = evaluate(test_dataset_padded, test_labels, batch_size, accuracy_operation, x, y_)
        test_accuracy = evaluate(test_dataset_padded, test_labels, batch_size, accuracy_operation, x, y_)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print("x", x)
        print("y_", y_)


        # a = tf.Print(x, [x], "this is x")
        # b = tf.multiply(a, 1)
        # a.eval()

    return [test_accuracy, ], protobuf, metagraph, x, y_, save_path  # do not remove coma, deap expect tuple rather than float










# this function is used exclusively to evaluate individual (get his fitness) by using test data and model extracted
# from protobuf
def evaluate_individual(individual, protobuf, metagraph, save_path, test_dataset, test_labels):
    # print(individual)
    tf.reset_default_graph()

    # extract genes from the individual
    batch_size, learning_rate, standard_dev, receptive_field_1, filters_1, receptive_field_2, \
        filters_2, activation = [x for x in individual]

    # here calculate padding depending on the size of the receptive field
    padding_size = int((receptive_field_1 - 1) / 2)

    # Pad images with 0s
    test_dataset_padded = padding(test_dataset, padding_size)

    x = tf.placeholder(tf.float32, shape=[None, (28 + padding_size * 2), (28 + padding_size * 2), 1])
    y_ = tf.placeholder(tf.int32, None)

    #x2 = tf.placeholder(tf.float32, shape=[None, (28 + padding_size * 2), (28 + padding_size * 2), 1])
    #y_2 = tf.placeholder(tf.int32, None)

    # with tf.Session() as sess:  print("x....", x)
    # with tf.Session() as sess:  print("y....", y_)
    # print("x+++", x)
    # print("y_+++", y_)


    # Invoke LeNet function by passing features
    logits = le_net_5(x, standard_dev, receptive_field_1, filters_1, receptive_field_2, filters_2, activation)

    # Softmax with cost function implementation
    # I think it does calculate error - measures difference between labels and output of the network
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)

    #loss_operation = tf.reduce_mean(cross_entropy)
    # Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
    # to update network weights iterative based in training data.
    # it is about changing weights to get better accuracy
    # optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # here I pass to the Adam calculated error (cross_entropy)
    #training_operation = optimizer.minimize(loss_operation)

    # here i compare results between between results from the network and labels
    # the tf.argmax is needed because results from network are probably not binary, I guess that
    # they come as probability and this tf.argmax just gets highest probability
    # not sure why we need argmax for labels... mystery...
    # it returns boolean tensor, something like 1 0 0 0 1 1 1 1 0 0 0 etc
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

    # here mean is calculated but for some unclear reason it needs input of float
    # maybe if it was not casted to float it would return mean as integer??
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # restrict how much gpu ram can be allocated to tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    #saver = tf.train.Saver.from_proto(protobuf, import_scope=None)
    # saver.restore(sess,)

    # define saver
    saver = tf.compat.v1.train.Saver()

    with tf.Session(config=config) as sess:
        # print("nodes of the graph: ", len(sess.graph._nodes_by_name.keys()))
        # tf.global_variables_initializer Returns an Op that initializes global variables.
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.import_meta_graph(metagraph)
        print("x", x)
        print("y_", y_)
        saver.restore(sess, save_path)
        print("x", x)
        print("y_", y_)
        # i guess this does not work, i see that almost all GPU ram is allocated
        # while it should allocate only minimum needed
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True


        #print("Model saved %s "%save_path)
        #protobuf = saver.to_proto(export_scope=None)
        test_accuracy = evaluate(test_dataset_padded, test_labels, batch_size, accuracy_operation, x, y_)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    return [test_accuracy, ]  # do not remove coma, deap expects tuple rather than float




# this function is used exclusively to evaluate individual (get his fitness) by using test data and model extracted
# from protobuf
def evaluate_individual_2(individual, protobuf, metagraph, x, y_, save_path, test_dataset, test_labels):
    # print(individual)
    tf.reset_default_graph()

    # extract genes from the individual
    batch_size, learning_rate, standard_dev, receptive_field_1, filters_1, receptive_field_2, \
        filters_2, activation = [x for x in individual]

    # here calculate padding depending on the size of the receptive field
    padding_size = int((receptive_field_1 - 1) / 2)

    # Pad images with 0s
    test_dataset_padded = padding(test_dataset, padding_size)

    #x = tf.placeholder(tf.float32, shape=[None, (28 + padding_size * 2), (28 + padding_size * 2), 1])
    #y_ = tf.placeholder(tf.int32, None)

    #x2 = tf.placeholder(tf.float32, shape=[None, (28 + padding_size * 2), (28 + padding_size * 2), 1])
    #y_2 = tf.placeholder(tf.int32, None)

    # with tf.Session() as sess:  print("x....", x)
    # with tf.Session() as sess:  print("y....", y_)
    # print("x+++", x)
    # print("y_+++", y_)


    # Invoke LeNet function by passing features
    logits = le_net_5(x, standard_dev, receptive_field_1, filters_1, receptive_field_2, filters_2, activation)

    # Softmax with cost function implementation
    # I think it does calculate error - measures difference between labels and output of the network
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)

    #loss_operation = tf.reduce_mean(cross_entropy)
    # Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
    # to update network weights iterative based in training data.
    # it is about changing weights to get better accuracy
    # optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # here I pass to the Adam calculated error (cross_entropy)
    #training_operation = optimizer.minimize(loss_operation)

    # here i compare results between between results from the network and labels
    # the tf.argmax is needed because results from network are probably not binary, I guess that
    # they come as probability and this tf.argmax just gets highest probability
    # not sure why we need argmax for labels... mystery...
    # it returns boolean tensor, something like 1 0 0 0 1 1 1 1 0 0 0 etc
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

    # here mean is calculated but for some unclear reason it needs input of float
    # maybe if it was not casted to float it would return mean as integer??
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # restrict how much gpu ram can be allocated to tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    #saver = tf.train.Saver.from_proto(protobuf, import_scope=None)
    # saver.restore(sess,)

    # define saver
    saver = tf.compat.v1.train.Saver()

    with tf.Session(config=config) as sess:
        # print("nodes of the graph: ", len(sess.graph._nodes_by_name.keys()))
        # tf.global_variables_initializer Returns an Op that initializes global variables.
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.import_meta_graph(metagraph)
        print("x", x)
        print("y_", y_)
        saver.restore(sess, save_path)
        print("x", x)
        print("y_", y_)
        # i guess this does not work, i see that almost all GPU ram is allocated
        # while it should allocate only minimum needed
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True


        #print("Model saved %s "%save_path)
        #protobuf = saver.to_proto(export_scope=None)
        test_accuracy = evaluate(test_dataset_padded, test_labels, batch_size, accuracy_operation, x, y_)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    return [test_accuracy, ]  # do not remove coma, deap expects tuple rather than float







# this function continues training individual using pretrained model
# this function should not duplicate code from the function to train from scratch (train individual)
# but rather reuse some code,
# an option would be to make protobuf an optional attribute, which when not paseed
# should indicate that individual should should be trained from scratch
def continue_training_individual (individual, protobuf, max_time, train_dataset, valid_dataset, test_dataset,
                     train_labels, valid_labels, test_labels):

    return [test_accuracy, ], protobuf  # do not remove coma, deap expects tuple rather than float
