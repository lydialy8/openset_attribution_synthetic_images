import sys

import yaml
from sklearn.metrics import accuracy_score
from sklearn import metrics
from dataset import *
from model import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import pandas as pd
from scipy.spatial.distance import cdist

####   DECLARATIONS   ####

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
nb_classes = configurations["NB_CLASSES_PER_BATCH"]
nb_imgs_per_class = configurations["NB_IMGS_PER_CLASS"]


model_path = configurations["MODEL_PATH"]
resume = configurations["RESUME"]

embeddingmodel_path = os.path.join(experiment_name, 'embedding_models', configurations["EMBEDDED_MODEL_NAME"])
numpy_path = os.path.join(experiment_name, 'variables')
os.makedirs(numpy_path, exist_ok=True)

closed_paths = configurations["PATHS"]
closed_classes = configurations["CLASSES"]
open_paths = configurations["OPEN_PATHS"]
open_classes = configurations["OPEN_CLASSES"]
network = configurations["NETWORK"]

classes = closed_classes + open_classes

nbofimgs = []
subfolders = {}
for c in classes:
    nbofimgs.extend([configurations[c + "_PARAMETERS"]["NBOFIMGS"]])
    subfolders[c] = configurations[f"{c}_PARAMETERS"]["SUBFOLDERS"]

embeddingmodel_path = os.path.join(experiment_name, 'embedding_models', configurations["EMBEDDED_MODEL_NAME"])
densemodel_path = os.path.join(experiment_name, 'densemodels', configurations["DENSE_MODEL_NAME"])

mirrored_strategy = tf.distribute.MirroredStrategy()

####   DATASETS   ####
data, true_labels = data_load_test(closed_paths, open_paths, experiment_name + '/', nbofimgs, subfolders, classes,
                                   testDense=True)

####   MODEL   ####
model = model_2denseLayers(img_res, emb_size, embeddingmodel_path, network)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights(densemodel_path)



### Getting Centroids ###
val_ds = data_generator(data[0], np.array(true_labels[0]).astype(int),img_res, batch_size)

net = embedding_model(img_res, emb_size, network)
net.load_weights(embeddingmodel_path)

x_val_embeddings = net.predict(val_ds)
labels_val = np.concatenate([x.ravel() for x in [y.numpy().astype(int) for x, y, z in val_ds]])
filepaths_val = np.concatenate([x.ravel() for x in [z.numpy() for x, y, z in val_ds]])

file_paths=[]
centroids = []
for i in range(len(closed_classes)):
    kmeans = KMeans(init="k-means++", n_clusters=1, random_state=0, n_init='auto').fit(
        x_val_embeddings[labels_val==i])

    min_dist = np.min(cdist(x_val_embeddings[labels_val==i], kmeans.cluster_centers_, 'euclidean'), axis=1)
    centroids.append(x_val_embeddings[labels_val==i][np.argmin(min_dist)])
    file_paths.append(np.array(filepaths_val)[labels_val==i][np.argmin(min_dist)])


####   RESULTS WITH FIVE FIXED REFERENCES FROM CLOSED SET ARCHITECTURE  ####
test_dataset = data_generator_pairs(data[1], np.array(true_labels[1]).astype(int), img_res, batch_size, file_paths=file_paths, all_classes=True)

test_lbls = np.concatenate([x.ravel() for x in [y["lbl"].numpy().astype(int) for x, y in test_dataset]])
input_lbls = np.concatenate([x.ravel() for x in [y["input_lbl"].numpy().astype(int) for x, y in test_dataset]])
ref_lbls = np.concatenate([x.ravel() for x in [y["ref_lbl"].numpy().astype(int) for x, y in test_dataset]])
img_idx = np.concatenate([x.ravel() for x in [y["img_idx"].numpy().astype(int) for x, y in test_dataset]])
logits = model.predict(test_dataset)
predictions = np.array(tf.keras.layers.Activation('sigmoid')(logits))
np.save(f'{numpy_path}/logits_5fixed.npy', logits)
np.save(f'{numpy_path}/predictions_5fixed.npy', predictions)
np.save(f'{numpy_path}/test_lbls_5fixed.npy', test_lbls)
np.save(f'{numpy_path}/ref_lbls_5fixed.npy', ref_lbls)
np.save(f'{numpy_path}/input_lbls_5fixed.npy', input_lbls)
np.save(f'{numpy_path}/img_idx_5fixed.npy', img_idx)


df = pd.DataFrame({'img_idx': img_idx, 'gt_lbl': test_lbls, 'input_lbl': input_lbls,
                   'ref_lbl': ref_lbls, 'predictions': predictions[:,0], 'logits': logits[:, 0]})


grouped_prediction_scores_mn = df.loc[df.groupby('img_idx').predictions.idxmin()].reset_index(drop=True)
grouped_logits_scores_mn = df.loc[df.groupby('img_idx').logits.idxmin()].reset_index(drop=True)


preds = np.array(grouped_prediction_scores_mn['predictions'].values)
gt_lbl = np.array(grouped_prediction_scores_mn['gt_lbl'].values)
input_lbl = np.array(grouped_prediction_scores_mn['input_lbl'].values)
ref_lbl = np.array(grouped_prediction_scores_mn['ref_lbl'].values)

prediction_lbls=np.array([1 if x!=y else 0 for x,y in zip(input_lbl,ref_lbl )])
predicted_lbls_closed = prediction_lbls[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
gt_lbls_closed = gt_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]

closed_acc = accuracy_score(gt_lbls_closed, predicted_lbls_closed).round(2)
print(f"closedset accuracy: {closed_acc}", flush=True)

fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, preds)
auc_predictions = metrics.auc(fpr, tpr)
print(f"auc_predictions: {auc_predictions}", flush=True)

logits = np.array(grouped_logits_scores_mn['logits'].values)
gt_lbl = np.array(grouped_logits_scores_mn['gt_lbl'].values)
input_lbl = np.array(grouped_logits_scores_mn['input_lbl'].values)
ref_lbl = np.array(grouped_logits_scores_mn['ref_lbl'].values)

prediction_lbls=np.array([1 if x!=y else 0 for x,y in zip(input_lbl,ref_lbl)])
predicted_lbls_closed = prediction_lbls[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
gt_lbls_closed = gt_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]

closed_acc = accuracy_score(gt_lbls_closed, predicted_lbls_closed).round(2)
print(f"closedset accuracy: {closed_acc}", flush=True)

fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, logits)
auc_predictions = metrics.auc(fpr, tpr)
print(f"auc_predictions: {auc_predictions}", flush=True)