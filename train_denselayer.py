import sys

import yaml

from dataset import *
from model import *

config_path = 'configurations/efficient_5arch_config3.yml'

if len(sys.argv) > 1:
    config_path = sys.argv[1]

configurations = yaml.safe_load(open(config_path))

emb_size = configurations["EMBEDDING_SIZE"]
img_res = configurations["IMAGE_RESOLUTION"]
epochs = configurations["EPOCHS"]
lr = configurations["LEARNING_RATE"]
experiment_name = configurations["EXPERIMENT_NAME"]
batch_size = configurations["BATCH_SIZE"]

model_path = configurations["MODEL_PATH"]
resume = configurations["RESUME"]
nb_classes = configurations["NB_CLASSES_PER_BATCH"]
nb_imgs_per_class = configurations["NB_IMGS_PER_CLASS"]

embeddingmodel_path = os.path.join(experiment_name, 'embedding_models',
                                   configurations["EMBEDDED_MODEL_NAME"])

paths = configurations["PATHS"]
classes = configurations["CLASSES"]
network = configurations["NETWORK"]

nbofimgs = []
subfolders = {}
for c in classes:
    nbofimgs.extend([configurations[c + "_PARAMETERS"]["NBOFIMGS"]])
    subfolders[c] = configurations[f"{c}_PARAMETERS"]["SUBFOLDERS"]

mirrored_strategy = tf.distribute.MirroredStrategy()

data, label = data_load_train(paths, experiment_name + '/', nbofimgs, subfolders, classes, predefined=True)
train_data, train_label = data[0], np.array(label[0]).astype(int)
valid_data, valid_label = data[1], np.array(label[1]).astype(int)

train_dataset = data_generator_pairs(train_data, train_label, img_res, batch_size, aug=True, epochs=epochs)
val_dataset = data_generator_pairs(valid_data, valid_label, img_res, batch_size)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=experiment_name + '/densemodels/' + experiment_name + '.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
os.makedirs(f'{experiment_name}/densemodels', exist_ok=True)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=5,
    decay_rate=0.96,
    staircase=True)  # lr = lr * (decay_rate ^ decay_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

with mirrored_strategy.scope():
    model = model_2denseLayers(img_res, emb_size, embeddingmodel_path, network)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])

if resume:
    model.load_weights(model_path)
    print("weights loaded")

model.layers[2].trainable = False
model.summary()
history = model.fit(
    train_dataset,
    epochs=epochs, verbose=1, validation_data=val_dataset
    , callbacks=my_callbacks)