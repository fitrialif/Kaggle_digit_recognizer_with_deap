import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import time
import numpy as np

def augment_data(images_dataset, labels_dataset, multiplier):

    # size (in pictures) of one job send to the child process to work on
    max_size_of_one_job = 30

    time_start = time.time()
    # some funny lambda function to randomly decide to make augmentation or not
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # we do following augmentations with probability of 50%
    # - translation in the plane xy by +-3 pixels
    # - scaling from 80% to 120%
    # - rotation by +- 35deg
    # - gaussian blur
    aug = iaa.Sequential([
        sometimes(iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)})),
        sometimes(iaa.Affine(scale=(0.80, 1.20))),
        sometimes(iaa.Affine(rotate=(-35, 35))),
        sometimes(iaa.GaussianBlur(sigma=(0.1, 0.5)))
    ])

    # calculate how many pieces of array with training data we can get using max number of pictures
    # that should be processed per one child process
    pieces = len(images_dataset) // max_size_of_one_job
    if pieces == 0:
        pieces = 1
    # print("before if check ", pieces)
    # if len(images_dataset) % (pieces * max_size_of_one_job) != 0:
    #     pieces = pieces + 1
    #     print("in if check ", pieces)

    # split training data into pieces
    split_training_dataset = np.array_split(images_dataset, pieces)
    split_training_labels = np.array_split(labels_dataset, pieces)

    batches = []
    # for each piece generate batches that will be augmented
    for i in range(pieces):
        batches = batches + [UnnormalizedBatch(images=split_training_dataset[i], data=split_training_labels[i]) for _ in
                             range(multiplier)]

    # run jobs in 32 child processes
    with aug.pool(processes=32, maxtasksperchild=200, seed=1) as pool:
        print("sending for augmentation batches : ", len(batches))
        batches_aug = pool.map_batches(batches)

    # concatenate all data back together
    all_images_dataset = np.concatenate((images_dataset, np.concatenate(([i.images_aug[:] for i in batches_aug]))))
    all_labels_dataset = np.concatenate((labels_dataset, np.concatenate(([i.data[:] for i in batches_aug]))))


    time_end = time.time()
    print("Augmentation done in %.2fs" % (time_end - time_start,))

    print(" len of all_images_dataset : ", len(all_images_dataset))
    print(" len of all_labels_dataset  : ", len(all_labels_dataset))

    print(" end ")
    return all_images_dataset, all_labels_dataset