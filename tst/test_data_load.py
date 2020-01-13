import unittest
import data_load
import deap


class BaseUnitTest(unittest.TestCase):
    def setUp(self):
        print(self._testMethodDoc)


class TestDataLoad(BaseUnitTest):
    def test_data_load(self):
        """testing data_load.data_load """
        input_file = '../data/sample.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        # just check if shape is correct
        self.assertEqual(training_dataset.shape, (80, 28, 28, 1))
        self.assertEqual(testing_dataset.shape, (10, 28, 28, 1))
        self.assertEqual(validating_dataset.shape, (10, 28, 28, 1))

        self.assertEqual(training_labels.shape, (80, 10))
        self.assertEqual(testing_labels.shape, (10, 10))
        self.assertEqual(validating_labels.shape, (10, 10))


    def test_data_load2(self):
        """testing data_load.data_load """
        input_file = '../data/sample.csv'
        input_file = '../data/train.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        from matplotlib import pyplot as plt
        import random
        random_picture = random.randint(0, len(training_dataset))
        print("random int : ", random_picture)

        plt.imshow(training_dataset[random_picture].reshape(28,28), interpolation='nearest')
        plt.show()
        print(training_labels[random_picture])

        import imgaug.augmenters as iaa
        # seq = iaa.Sequential([
        #     iaa.Crop(px=(1, 16), keep_size=False),
        #     iaa.Fliplr(0.5),
        #     iaa.GaussianBlur(sigma=(0, 3.0))])

        # for i in range(10):
        #     seq = iaa.Affine(translate_px=(-3, 3))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()



        # for i in range(10):
        #     seq = iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)})
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        # for i in range(10):
        #     print("i : ", i)
        #     seq = iaa.Affine(shear=(-16, 16))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        # for i in range(10):
        #     print("i : ", i)
        #     seq = iaa.Affine(scale=(0.80, 1.20))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        #
        # for i in range(10):
        #     print("i : ", i)
        #     seq = iaa.Affine(rotate=(-35, 35))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        #import imgaug as ia
        import time
        import numpy as np
        from imgaug.augmentables.batches import UnnormalizedBatch
        from imgaug import multicore
        for i in range(1):
            #print("i : ", i)

            BATCH_SIZE = 16
            NB_BATCHES = 9

            batches = [UnnormalizedBatch(images=training_dataset, data=training_labels) for _ in range(NB_BATCHES)]

            #seq = iaa.GaussianBlur(sigma=(0.1, 0.5))


            aug = iaa.Sequential([

                #iaa.Affine(translate_px=(-3, 3))
                sometimes(iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)})),
                sometimes(iaa.Affine(scale=(0.80, 1.20))),
                sometimes(iaa.Affine(rotate=(-35, 35))),
                sometimes(iaa.GaussianBlur(sigma=(0.1, 0.5)))

                # iaa.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
                # iaa.Fliplr(0.5),  # very fast
                # iaa.CropAndPad(px=(-10, 10))  # very fast
            ])
            #new_images = seq(images=training_dataset)
            # plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
            # plt.show()
            #ia.imshow(new_images[70].reshape(28, 28))

            time_start = time.time()
            #batches_aug = list(seq.augment_batches(batches, background=True))

            with aug.pool(processes=32, maxtasksperchild=200, seed=1) as pool:
                batches_aug = pool.map_batches(batches)

            for i in range(NB_BATCHES):

                plt.imshow(batches_aug[i].images_aug[random_picture].reshape(28, 28), interpolation='nearest')
                plt.show()

                print(batches_aug[i].data[random_picture])

            # plt.imshow(batches_aug[0].images_aug[random_picture].reshape(28, 28), interpolation='nearest')
            # plt.show()

            time_end = time.time()
            print("Augmentation done in %.2fs" % (time_end - time_start,))

        all_training_dataset = training_dataset
        all_training_labels = training_labels
        for i in range(NB_BATCHES):
            all_training_dataset = np.concatenate((all_training_dataset, batches_aug[i].images_aug))
            all_training_labels = np.concatenate((all_training_labels, batches_aug[i].data))

        print(" len of all_training_dataset : ", len(all_training_dataset))
        print(" len of all_training_labels  : ", len(all_training_labels))

        print(" end ")



    def test_data_load4(self):
        """testing data_load.data_load """
        input_file = '../data/sample.csv'
        input_file = '../data/train.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        from matplotlib import pyplot as plt
        import random
        random_picture = random.randint(0, len(training_dataset))
        print("random int : ", random_picture)

        plt.imshow(training_dataset[random_picture].reshape(28,28), interpolation='nearest')
        plt.show()
        print(training_labels[random_picture])

        import imgaug.augmenters as iaa

        import time
        import numpy as np
        from imgaug.augmentables.batches import UnnormalizedBatch
        #from imgaug import multicore
        #import imgaug as ia

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        #BATCH_SIZE = 16
        NB_BATCHES = 200
        #max_size_of_one_job = 301
        max_size_of_one_job = 300

        #seq = iaa.GaussianBlur(sigma=(0.1, 0.5))


        aug = iaa.Sequential([
            sometimes(iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)})),
            sometimes(iaa.Affine(scale=(0.80, 1.20))),
            sometimes(iaa.Affine(rotate=(-35, 35))),
            sometimes(iaa.GaussianBlur(sigma=(0.1, 0.5)))
        ])

        time_start = time.time()

        #all_training_dataset = training_dataset
        #all_training_labels = training_labels

        #print("initial size of the all_training_dataset : ", len(all_training_dataset))

        with aug.pool(processes=32, maxtasksperchild=200, seed=1) as pool:


            # calculate how many pieces of array with training data we can get using max number of pictures
            # that should be processed per one child process
            pieces = len(training_dataset) // max_size_of_one_job
            print("before if check ", pieces)
            if len(training_dataset)%(pieces*max_size_of_one_job) != 0:
                pieces = pieces + 1
                print("in if check ", pieces)

            # split training data into pieces
            split_training_dataset = np.array_split(training_dataset, pieces)
            split_training_labels = np.array_split(training_labels, pieces)

            batches = []
            # for each piece generate batches that will be augmented
            for i in range(pieces):
                batches = batches + [UnnormalizedBatch(images=split_training_dataset[i], data=split_training_labels[i]) for _ in
                           range(NB_BATCHES)]

            # call sending tasks to children
            print("sending for augmentation batches : ", len(batches))
            batches_aug = pool.map_batches(batches)

            # join together all augmented sets (including original pictures)
            print("joining all the batches with original pictures")
            #for i in range(len(batches_aug)):
                #print(i)
                #all_training_dataset = np.concatenate((all_training_dataset, batches_aug[i].images_aug))
                #all_training_labels = np.concatenate((all_training_labels, batches_aug[i].data))

            #x = (i.images_aug[:] for i in batches_aug)
            #y = np.concatenate(x)

            all_training_dataset = np.concatenate((training_dataset, np.concatenate(([i.images_aug[:] for i in batches_aug]))))
            #all_training_dataset = np.concatenate((all_training_dataset, (i.images_aug for i in batches_aug)) )
            all_training_labels = np.concatenate((training_labels, np.concatenate(([i.data[:] for i in batches_aug]))))

                #print("intermediate size of the all_training_dataset : ", len(all_training_dataset))
        print("final size of the all_training_dataset : ", len(all_training_dataset))

        # print some random pictures
        #import matplotlib.pyplot as plt
        plt.imshow(all_training_dataset[79].reshape(28, 28), interpolation='nearest')
        plt.show()
        print(all_training_labels[79])

        plt.imshow(all_training_dataset[81].reshape(28, 28), interpolation='nearest')
        plt.show()
        print(all_training_labels[81])
        print("----------------")
        for i in range(10):
            random_picture = random.randint(0, len(all_training_dataset))
            plt.imshow(all_training_dataset[random_picture].reshape(28, 28), interpolation='nearest')
            plt.show()

            print(all_training_labels[random_picture])

        # plt.imshow(batches_aug[0].images_aug[random_picture].reshape(28, 28), interpolation='nearest')
        # plt.show()

        time_end = time.time()
        print("Augmentation done in %.2fs" % (time_end - time_start,))

        print(" len of all_training_dataset : ", len(all_training_dataset))
        print(" len of all_training_labels  : ", len(all_training_labels))

        print(" end ")



    def test_data_load3(self):
        """testing data_load.data_load """
        input_file = '../data/sample.csv'
        #input_file = '../data/train.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        from matplotlib import pyplot as plt
        plt.imshow(training_dataset[70].reshape(28,28), interpolation='nearest')
        plt.show()

        import imgaug.augmenters as iaa
        # seq = iaa.Sequential([
        #     iaa.Crop(px=(1, 16), keep_size=False),
        #     iaa.Fliplr(0.5),
        #     iaa.GaussianBlur(sigma=(0, 3.0))])

        # for i in range(10):
        #     seq = iaa.Affine(translate_px=(-3, 3))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()



        # for i in range(10):
        #     seq = iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)})
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[61].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        # for i in range(10):
        #     print("i : ", i)
        #     seq = iaa.Affine(shear=(-16, 16))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        # for i in range(10):
        #     print("i : ", i)
        #     seq = iaa.Affine(scale=(0.80, 1.20))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()


        # for i in range(10):
        #     print("i : ", i)
        #     seq = iaa.Affine(rotate=(-35, 35))
        #     new_images = seq(images=training_dataset)
        #     plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     plt.show()

        # #import imgaug as ia
        # import time
        # from imgaug.augmentables.batches import UnnormalizedBatch
        # from imgaug import multicore
        # for i in range(1):
        #     #print("i : ", i)
        #
        #     BATCH_SIZE = 16
        #     NB_BATCHES = 320
        #
        #     batches = [UnnormalizedBatch(images=training_dataset) for _ in range(NB_BATCHES)]
        #
        #     #seq = iaa.GaussianBlur(sigma=(0.1, 0.5))
        #     aug = iaa.Sequential([
        #
        #         #iaa.Affine(translate_px=(-3, 3))
        #         iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)}),
        #         iaa.Affine(scale=(0.80, 1.20)),
        #         iaa.Affine(rotate=(-35, 35)),
        #         iaa.GaussianBlur(sigma=(0.1, 0.5))
        #
        #         # iaa.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
        #         # iaa.Fliplr(0.5),  # very fast
        #         # iaa.CropAndPad(px=(-10, 10))  # very fast
        #     ])
        #     #new_images = seq(images=training_dataset)
        #     # plt.imshow(new_images[70].reshape(28, 28), interpolation='nearest')
        #     # plt.show()
        #     #ia.imshow(new_images[70].reshape(28, 28))
        #
        #     time_start = time.time()
        #     #batches_aug = list(seq.augment_batches(batches, background=True))
        #
        #     with aug.pool(processes=32, maxtasksperchild=200, seed=1) as pool:
        #         batches_aug = pool.map_batches(batches)
        #
        #     time_end = time.time()
        #     print("Augmentation done in %.2fs" % (time_end - time_start,))







class TestLoadInitialPopulation(BaseUnitTest):
    def test_load_initial_population(self):
        """testing data_load.load_initial_population"""

        input_file = '../data/initial_population2.txt'
        pop = data_load.load_initial_population(input_file)

        self.assertIsInstance(pop[0][0], int)
        self.assertIsInstance(pop[1][0], int)
        self.assertIsInstance(pop[0][1], float)
        self.assertIsInstance(pop[1][1], float)
        self.assertIsInstance(pop[0][7], int)
        self.assertIsInstance(pop[1][7], int)
        self.assertIsNotNone(pop[0].fitness)
        self.assertIsNotNone(pop[1].fitness)
        self.assertIsInstance(pop[0].fitness, deap.creator.FitnessMax)
        self.assertIsInstance(pop[1].fitness, deap.creator.FitnessMax)

    def test_load_initial_population_evaluate_few(self):
        """testing data_load.load_initial_population by evaluating few of them"""
        print()





class TestImageAugmentation(BaseUnitTest):
    def test_data_augmentation(self):
        """testing augment_data.augment_data"""

        input_file = '../data/sample.csv'
        #input_file = '../data/train.csv'
        image_size = 28
        multiplier = 10 # this is how many more data we would like to get through augmentation
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load2(input_file, image_size, multiplier)

        # just check if shape is correct
        self.assertEqual(training_dataset.shape, ((multiplier+1)*80, 28, 28, 1))
        self.assertEqual(testing_dataset.shape, ((multiplier+1)*10, 28, 28, 1))
        self.assertEqual(validating_dataset.shape, (10, 28, 28, 1))

        self.assertEqual(training_labels.shape, ((multiplier+1)*80, 10))
        self.assertEqual(testing_labels.shape, ((multiplier+1)*10, 10))
        self.assertEqual(validating_labels.shape, (10, 10))