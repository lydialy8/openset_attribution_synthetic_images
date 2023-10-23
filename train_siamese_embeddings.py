import sys

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import yaml
from sklearn.decomposition import PCA

from dataset import *
from model import *

mirrored_strategy = tf.distribute.MirroredStrategy()
config_path = 'configurations/efficient_5arch_config3.yml'

if len(sys.argv) > 1:
    config_path = sys.argv[1]

configurations = yaml.safe_load(open(config_path))

emb_size = configurations["EMBEDDING_SIZE"]
img_res = configurations["IMAGE_RESOLUTION"]
epochs = configurations["EPOCHS"]
lr = configurations["LEARNING_RATE"]
experiment_name = configurations["EXPERIMENT_NAME"]
model_path = configurations["MODEL_PATH"]
resume = configurations["RESUME"]
nb_classes = configurations["NB_CLASSES_PER_BATCH"]
nb_imgs_per_class = configurations["NB_IMGS_PER_CLASS"]

paths = configurations["PATHS"]
classes = configurations["CLASSES"]
network = configurations["NETWORK"]

nbofimgs = []
subfolders = {}
for c in classes:
    nbofimgs.extend([configurations[c + "_PARAMETERS"]["NBOFIMGS"]])
    subfolders[c] = configurations[f"{c}_PARAMETERS"]["SUBFOLDERS"]

data, label = data_load_train(paths, experiment_name, nbofimgs, subfolders, classes)

train_ds = balanced_image_dataset_from_directory(
    data[0], num_classes_per_batch=nb_classes,
    num_images_per_class=nb_imgs_per_class, labels=label[0], augmentation_enabled=True, image_size=(img_res, img_res),
    seed=555,
    safe_triplet=True)

val_ds = balanced_image_dataset_from_directory(
    data[1], num_classes_per_batch=nb_classes,
    num_images_per_class=nb_imgs_per_class, labels=label[1], image_size=(img_res, img_res), seed=555,
    safe_triplet=True)

test_ds = balanced_image_dataset_from_directory(
    data[2], num_classes_per_batch=nb_classes,
    num_images_per_class=nb_imgs_per_class, labels=label[2], image_size=(img_res, img_res), seed=555,
    safe_triplet=True)


class PCAPlotter(tf.keras.callbacks.Callback):

    def __init__(self, plt, embedding_model, test_dataset):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.dataset = test_dataset
        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        plt.ion()

        self.losses = []

    def plot(self, epoch=None, plot_loss=False):
        x_test_embeddings = self.embedding_model.predict_generator(self.dataset)
        pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
        labels = np.array([y.numpy().astype(int) for x, y in val_ds]).flatten()
        colors = np.array(['black', 'green', 'red', 'blue', 'violet'])
        self.ax1.clear()
        self.ax1.scatter(pca_out[:, 0], pca_out[:, 1], c=colors[labels], cmap='seismic')
        if plot_loss:
            self.ax2.clear()
            self.ax2.plot(range(epoch), self.losses)
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
        self.fig.canvas.draw()
        self.fig.savefig(f'{experiment_name}/embedding_scatter_plots_test_ds/test_latent_{epoch}.png')

    def on_train_begin(self, logs=None):
        self.losses = []
        self.fig.show()
        self.fig.canvas.draw()
        self.plot()

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plot(epoch + 1, plot_loss=True)


with mirrored_strategy.scope():
    net = embedding_model(img_res, emb_size, network)
    net.summary()
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=5,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    net.compile(loss=tfa.losses.ContrastiveLoss(), optimizer=optimizer) #TripletSemiHardLoss

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{experiment_name}/embedding_models/{experiment_name}' + '.{epoch:02d}-{val_loss:.4f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        PCAPlotter(plt, net, test_ds)
    ]

if resume:
    net.load_weights(model_path)
    print("weights loaded")

os.makedirs(f'{experiment_name}/embedding_models', exist_ok=True)
os.makedirs(f'{experiment_name}/embedding_scatter_plots_test_ds', exist_ok=True)
history = net.fit(train_ds, epochs=epochs, verbose=1, validation_data=val_ds, callbacks=my_callbacks)
