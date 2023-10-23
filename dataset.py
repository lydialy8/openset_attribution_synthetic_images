import os
import random
from functools import partial
from glob import glob

from balanced_batch import *


@tf.function
def getImage(path, img_width, aug):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_width, img_width))
    img.set_shape((img_width, img_width, 3))
    img = tf.cast(img, tf.float32) / 255
    if aug:
        img = augmentation(img)
    return img


def parse_path_test(img_width, record, lbl):
    img = getImage(record, img_width, False)
    return img, lbl, record


def parse_path(img_width, aug, all_classes, record, lbl, input_lbl, ref_lbl, img_idx):
    img1 = getImage(record[0], img_width, aug)
    img2 = getImage(record[1], img_width, aug)
    if all_classes:
        return {"input_1": img1, "input_2": img2}, {"lbl": lbl, "input_lbl": input_lbl, "ref_lbl": ref_lbl,
                                                    "img_idx": img_idx}
    return {"input_1": img1, "input_2": img2}, lbl


def data_generator(images, labels, img_width, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(partial(parse_path_test, img_width), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset


def data_generator_pairs(images, labels, img_width, batch_size, aug=False, epochs=None, all_classes=False,
                         nb_of_refs=1, file_paths=None):
    pairs, labels, input_class_label, ref_class_label, img_idx = make_pairs(images, labels, all_classes, nb_of_refs,
                                                                            file_paths)
    dataset = tf.data.Dataset.from_tensor_slices((pairs, labels, input_class_label, ref_class_label, img_idx))
    if epochs:
        dataset = dataset.shuffle(len(images), reshuffle_each_iteration=True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(partial(parse_path, img_width, aug, all_classes), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset


def make_pairs(x, y, all_classes, nb_of_refs, file_paths):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = int(max(y) + 1)
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    classes_lbls = np.array(range(num_classes))
    pairs = []
    labels = []
    input_class_label = []
    ref_class_label = []
    img_idx = []
    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        if file_paths:
            for idx2, x2 in enumerate(file_paths):
                pairs += [[x2, x1]]
                labels += [0 if idx2 == label1 else 1]
                input_class_label += [idx2]
                ref_class_label += [label1]
                img_idx += [idx1]
            """for label2 in range(5):
                idx2 = random.choice(digit_indices[label2])
                pairs += [[x[idx2], x1 ]]
                labels += [0 if label2 == label1 else 1]
                input_class_label += [label2]
                ref_class_label += [label1]
                img_idx += [idx1]"""

        else:
            for ref in range(nb_of_refs):
                idx2 = random.choice(digit_indices[label1])
                x2 = x[idx2]

                pairs += [[x1, x2]]
                labels += [0]
                input_class_label += [label1]
                ref_class_label += [label1]
                img_idx += [idx1]

            if all_classes:
                modified_classes = np.delete(classes_lbls, np.where(classes_lbls == label1))
                for label2 in modified_classes:
                    for ref in range(nb_of_refs):
                        idx2 = random.choice(digit_indices[label2])
                        x2 = x[idx2]
                        pairs += [[x1, x2]]
                        labels += [1]
                        input_class_label += [label1]
                        ref_class_label += [label2]
                        img_idx += [idx1]
            else:
                label2 = random.randint(0, num_classes - 1)
                while label2 == label1:
                    label2 = random.randint(0, num_classes - 1)
                idx2 = random.choice(digit_indices[label2])
                x2 = x[idx2]
                input_class_label += [label1]
                ref_class_label += [label2]
                img_idx += [idx1]
                pairs += [[x1, x2]]
                labels += [1]

    return np.array(pairs), np.array(labels), np.array(input_class_label), np.array(ref_class_label), np.array(img_idx)


def list_imgs(parents, config_dir, model, nbofimgs, subfolders, predefined=False):
    '''
    parents: the path for training images, under the path, it should contain several folders
    training_num: indicate the number of the images per architecture for training
    pre_defined: should be a txt file of the images for training, valid, and test; if None, then
                randomly read images
    models: the architecture name

    return the list of training, valid, and test images
    '''
    train_samples = []
    valid_samples = []
    test_samples = []

    if not predefined:
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        for i, folder in enumerate(subfolders):
            img_lists = sum([glob(os.path.join(parents, folder, '*.' + x)) for x in ['jpg', 'jpeg', 'png']], list())
            rand_perm = np.random.permutation(np.arange(len(img_lists)))
            img_lists = [img_lists[x] for x in rand_perm]
            sampled = img_lists
            if nbofimgs == 0:
                end_images = 2500
            else:
                end_images = nbofimgs
            train_samples += sampled[:int(nbofimgs * 0.9)]
            valid_samples += sampled[int(nbofimgs * 0.9):int(nbofimgs * 0.95)]
            test_samples += sampled[int(nbofimgs * 0.95):int(end_images * 1)]

        with open(f'{config_dir}/{model}_train_list.txt', 'w') as f:
            for i in (train_samples + valid_samples + test_samples):
                f.write(i + '\n')
    else:
        file_name = config_dir + model + '_train_list.txt'
        img_list = list()
        with open(file_name) as class_file:
            for line in class_file.readlines():
                img_name = line[:-1]
                img_list.append(img_name)
        training_num = len(img_list)
        train_samples = img_list[:int(training_num * 0.9)]
        valid_samples = img_list[int(training_num * 0.9):int(training_num * 0.95)]
        test_samples = img_list[int(training_num * 0.95):]
    return [train_samples, valid_samples, test_samples]


def data_load_train(paths, config_dir, nbofimgs, subfolders, classes, predefined=False):
    '''
    Path should be a list
    return all the training samples in a list
    '''
    train_samples = []
    valid_samples = []
    test_samples = []
    train_labels = []
    valid_labels = []
    test_labels = []
    for i, path in enumerate(paths):
        samples = list_imgs(path, config_dir, classes[i], nbofimgs[i], subfolders[classes[i]], predefined=predefined)
        print(f'{i} train:{len(samples[0])}')
        print(f'{i} val:{len(samples[1])}')
        print(f'{i} test:{len(samples[2])}')
        train_samples += samples[0]
        train_labels += list(np.ones(len(samples[0])) * i)
        valid_samples += samples[1]
        valid_labels += list(np.ones(len(samples[1])) * i)
        test_samples += samples[2]
        test_labels += list(np.ones(len(samples[2])) * i)
        del samples
    return [train_samples, valid_samples, test_samples], [train_labels, valid_labels, test_labels]


def data_load_test(closed_paths, open_paths, config_dir, nbofimgs, subfolders, classes, subfolders_open={},
                   predefined=True,
                   testDense=False):
    '''
    Path should be a list
    return all the training samples in a list
    '''
    valid_samples = []
    test_samples = []
    valid_labels = []
    test_labels = []
    os.makedirs(os.path.join(config_dir, 'cross'), exist_ok=True)
    for i, path in enumerate(closed_paths):
        if classes[i] in subfolders_open:
            samples = list_imgs(path, os.path.join(config_dir, 'cross'), classes[i], 0, subfolders_open[classes[i]],
                                predefined=False)
        else:
            samples = list_imgs(path, config_dir, classes[i], nbofimgs[i], subfolders[classes[i]],
                                predefined=predefined)
        print(f'{i} val:{len(samples[1])}', flush=True)
        print(f'{i} test:{len(samples[2])}', flush=True)
        valid_samples += samples[1]
        valid_labels += list(np.ones(len(samples[1])) * i)
        test_samples += samples[2][:500]
        test_labels += list(np.ones(len(samples[2])) * i)[:500]  # [classes[i]]
        del samples

    strt_idx = len(closed_paths)
    if open_paths:
        for i, path in enumerate(open_paths):
            j = strt_idx + i
            if classes[j] in subfolders_open:
                samples = list_imgs(path, config_dir, classes[j], 0, subfolders_open[classes[j]], predefined=False)
            else:
                samples = list_imgs(path, config_dir, classes[j], nbofimgs[j], subfolders[classes[j]], predefined=False)
            print(f'{j} test:{len(samples[2])}', flush=True)
            ### this only for test dense layers for latent space need to be set to False ###
            if testDense:
                print(f'{j} val:{len(samples[1])}', flush=True)
                valid_samples += samples[1]
                valid_labels += list(np.ones(len(samples[1])) * j)
            ####################################################
            test_samples += samples[2]
            test_labels += list(np.ones(len(samples[2])) * j) # list(len(samples[2]) * j)
            del samples

    return [valid_samples, test_samples], [valid_labels, test_labels]
