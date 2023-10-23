import sys

import yaml
from sklearn.metrics import accuracy_score
from sklearn import metrics
from dataset import *
from model import *

import pandas as pd

####   DECLARATIONS   ####

config_path = 'configurations/swin_5arch_config1.yml'

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

####   RESULTS WITH ONE REFERENCE   ####
test_dataset = data_generator_pairs(data[1], np.array(true_labels[1]).astype(int), img_res, batch_size, all_classes=True)

predictions = np.array([])

test_lbls = np.concatenate([x.ravel() for x in [y["lbl"].numpy().astype(int) for x, y in test_dataset]])
input_lbls = np.concatenate([x.ravel() for x in [y["input_lbl"].numpy().astype(int) for x, y in test_dataset]])
ref_lbls = np.concatenate([x.ravel() for x in [y["ref_lbl"].numpy().astype(int) for x, y in test_dataset]])

logits = model.predict(test_dataset)
predictions = np.append(predictions, tf.keras.layers.Activation('sigmoid')(logits))
np.save(f'{numpy_path}/logits.npy', logits)
np.save(f'{numpy_path}/predictions.npy', predictions)
np.save(f'{numpy_path}/test_lbls.npy', test_lbls)
np.save(f'{numpy_path}/input_lbls.npy', input_lbls)
np.save(f'{numpy_path}/ref_lbls.npy', ref_lbls)

isMatching_lbl = np.array(test_lbls)
predicted_label = np.array([1 if score > 0.5 else 0 for score in predictions])
acc = accuracy_score(isMatching_lbl, predicted_label).round(2)

fpr, tpr, thresholds = metrics.roc_curve(isMatching_lbl, predictions)
auc_predictions = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(isMatching_lbl, logits)
auc_logits = metrics.auc(fpr, tpr)
print(f"overall accuracy: {acc}", flush=True)
print(f"overall auc_predictions: {auc_predictions}", flush=True)
print(f"overall auc_logits: {auc_logits}", flush=True)

logits_open=logits[((input_lbls > len(closed_classes) - 1) & (ref_lbls > len(closed_classes) - 1))]
predictons_open=predictions[((input_lbls > len(closed_classes) - 1) & (ref_lbls > len(closed_classes) - 1))]
predicted_lbls_open = predicted_label[((input_lbls > len(closed_classes) - 1) & (ref_lbls > len(closed_classes) - 1))]
gt_lbls_open = isMatching_lbl[((input_lbls > len(closed_classes) - 1) & (ref_lbls > len(closed_classes) - 1))]

open_acc = accuracy_score(gt_lbls_open, predicted_lbls_open).round(2)
fpr, tpr, thresholds = metrics.roc_curve(gt_lbls_open, predictons_open)
auc_predictions = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(gt_lbls_open, logits_open)
auc_logits = metrics.auc(fpr, tpr)

print(f"openset accuracy: {open_acc}", flush=True)
print(f"openset auc_predictions: {auc_predictions}", flush=True)
print(f"openset auc_logits: {auc_logits}", flush=True)

logits_closed=logits[((input_lbls < len(closed_classes)) & (ref_lbls < len(closed_classes)))]
predictons_closed=predictions[((input_lbls < len(closed_classes)) & (ref_lbls < len(closed_classes)))]
predicted_lbls_closed = predicted_label[((input_lbls < len(closed_classes)) & (ref_lbls < len(closed_classes)))]
gt_lbls_closed = isMatching_lbl[((input_lbls < len(closed_classes)) & (ref_lbls < len(closed_classes)))]

closed_acc = accuracy_score(gt_lbls_closed, predicted_lbls_closed).round(2)
fpr, tpr, thresholds = metrics.roc_curve(gt_lbls_closed, predictons_closed)
auc_predictions = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(gt_lbls_closed, logits_closed)
auc_logits = metrics.auc(fpr, tpr)

print(f"closedset accuracy: {closed_acc}", flush=True)
print(f"closedset auc_predictions: {auc_predictions}", flush=True)
print(f"closedset auc_logits: {auc_logits}", flush=True)

for input in range(len(classes)):
    gt_lbl = isMatching_lbl[(input_lbls == input)]
    pd_lbl = predicted_label[(input_lbls == input)]
    pds= predictions[(input_lbls == input)]
    lts= logits[(input_lbls == input)]
    fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, pds)
    auc_predictions = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, lts)
    auc_logits = metrics.auc(fpr, tpr)
    print(f'input: {input}, acc: {accuracy_score(gt_lbl, pd_lbl).round(2)}, auc_predictions: {auc_predictions}, auc_logits: {auc_logits}', flush=True)
    for ref in range(len(classes)):
        gt_lbl=isMatching_lbl[(input_lbls == input) & (ref_lbls == ref)]
        pd_lbl=predicted_label[(input_lbls == input) & (ref_lbls == ref)]
        print(
            f'input: {input}, ref: {ref}, acc: {accuracy_score(gt_lbl, pd_lbl).round(2)}', flush=True)


####   RESULTS WITH MULTIPLE REFERENCES   ####
nb_samples = [100]
for n in nb_samples:
    test_dataset = data_generator_pairs(data[1], np.array(true_labels[1]).astype(int), img_res, batch_size, all_classes=True, nb_of_refs=n)
    predictions = np.array([])
    test_lbls = np.concatenate([x.ravel() for x in [y["lbl"].numpy().astype(int) for x, y in test_dataset]])
    input_lbls = np.concatenate([x.ravel() for x in [y["input_lbl"].numpy().astype(int) for x, y in test_dataset]])
    ref_lbls = np.concatenate([x.ravel() for x in [y["ref_lbl"].numpy().astype(int) for x, y in test_dataset]])
    img_idx = np.concatenate([x.ravel() for x in [y["img_idx"].numpy().astype(int) for x, y in test_dataset]])
    logits = model.predict(test_dataset)
    predictions = np.append(predictions, tf.keras.layers.Activation('sigmoid')(logits))
    np.save(f'{numpy_path}/logits_{n}.npy', logits)
    np.save(f'{numpy_path}/predictions_{n}.npy', predictions)
    np.save(f'{numpy_path}/test_lbls_{n}.npy', test_lbls)
    np.save(f'{numpy_path}/ref_lbls_{n}.npy', ref_lbls)
    np.save(f'{numpy_path}/input_lbls_{n}.npy', input_lbls)
    np.save(f'{numpy_path}/img_idx_{n}.npy', img_idx)

    isMatching_lbl = np.array(test_lbls)
    predicted_label = np.array([1 if score > 0.5 else 0 for score in predictions])
    df = pd.DataFrame({'img_idx':img_idx, 'prediction_lbl': predicted_label, 'gt_lbl': test_lbls, 'input_lbl':input_lbls, 'ref_lbl': ref_lbls, 'predictions': predictions, 'logits': logits[:,0]})
    grouped_prediction_scores_mean = df.groupby(['img_idx', 'gt_lbl', 'input_lbl', 'ref_lbl'])['predictions'].agg(np.mean).to_frame()
    grouped_prediction_scores_mn = df.groupby(['img_idx', 'gt_lbl', 'input_lbl', 'ref_lbl'])['predictions'].agg(np.min).to_frame()
    grouped_prediction_scores_mx = df.groupby(['img_idx', 'gt_lbl', 'input_lbl', 'ref_lbl'])['predictions'].agg(np.max).to_frame()
    """pred_lbl = np.array(grouped_predictions['prediction_lbl'].values)
    gt_lbl = np.array(grouped_predictions.prediction_lbl.index.get_level_values(level='gt_lbl'))
    input_lbl = np.array(grouped_predictions.prediction_lbl.index.get_level_values(level='input_lbl'))
    ref_lbl = np.array(grouped_predictions.prediction_lbl.index.get_level_values(level='ref_lbl'))
    acc = accuracy_score(gt_lbl, pred_lbl).round(2)
    open_gt_lbl=gt_lbl[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    open_pred_lbl=pred_lbl[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    open_acc = accuracy_score(open_gt_lbl,open_pred_lbl).round(2)
    closed_gt_lbl=gt_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
    closed_pred_lbl=pred_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
    closed_acc = accuracy_score(closed_gt_lbl,closed_pred_lbl).round(2)
    print(f"overall accuracy: {acc}", flush=True)
    print(f"openset accuracy: {open_acc}", flush=True)
    print(f"closedset accuracy: {closed_acc}", flush=True)

    for input in range(len(classes)):
        for ref in range(len(classes)):
            print( f'input: {input}, ref: {ref}, acc: {accuracy_score(gt_lbl[(input_lbl == input) & (ref_lbl == ref)], pred_lbl[(input_lbl == input) & (ref_lbl == ref)]).round(2)}', flush=True)"""


    preds = np.array(grouped_prediction_scores_mean['predictions'].values)
    gt_lbl = np.array(grouped_prediction_scores_mean.predictions.index.get_level_values(level='gt_lbl'))
    input_lbl = np.array(grouped_prediction_scores_mean.predictions.index.get_level_values(level='input_lbl'))
    ref_lbl = np.array(grouped_prediction_scores_mean.predictions.index.get_level_values(level='ref_lbl'))

    fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"overall auc_predictions mean: {auc_predictions}", flush=True)
    open_preds = preds[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    open_gt_lbl = gt_lbl[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    fpr, tpr, thresholds = metrics.roc_curve(open_gt_lbl, open_preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"openset auc_predictions mean: {auc_predictions}", flush=True)
    closed_preds=preds[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
    closed_gt_lbl=gt_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]

    fpr, tpr, thresholds = metrics.roc_curve(closed_gt_lbl, closed_preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"closedset auc_predictions mean: {auc_predictions}", flush=True)
    for input in range(len(classes)):
        gt_lbl_input = gt_lbl[(input_lbl == input)]
        pds = preds[(input_lbl == input)]
        fpr, tpr, thresholds = metrics.roc_curve(gt_lbl_input, pds)
        auc_predictions = metrics.auc(fpr, tpr)
        print(
            f'input: {input}, auc_predictions mean: {auc_predictions}',flush=True)
    preds = np.array(grouped_prediction_scores_mx['predictions'].values)
    gt_lbl = np.array(grouped_prediction_scores_mx.predictions.index.get_level_values(level='gt_lbl'))
    input_lbl = np.array(grouped_prediction_scores_mx.predictions.index.get_level_values(level='input_lbl'))
    ref_lbl = np.array(grouped_prediction_scores_mx.predictions.index.get_level_values(level='ref_lbl'))

    fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"overall auc_predictions max: {auc_predictions}", flush=True)
    open_preds = preds[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    open_gt_lbl = gt_lbl[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    fpr, tpr, thresholds = metrics.roc_curve(open_gt_lbl, open_preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"openset auc_predictions max: {auc_predictions}", flush=True)
    closed_preds=preds[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
    closed_gt_lbl=gt_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]

    fpr, tpr, thresholds = metrics.roc_curve(closed_gt_lbl, closed_preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"closedset auc_predictions max: {auc_predictions}", flush=True)
    for input in range(len(classes)):
        gt_lbl_input = gt_lbl[(input_lbl == input)]
        pds = preds[(input_lbl == input)]
        fpr, tpr, thresholds = metrics.roc_curve(gt_lbl_input, pds)
        auc_predictions = metrics.auc(fpr, tpr)
        print(
            f'input: {input}, auc_predictions max: {auc_predictions}',flush=True)
    preds = np.array(grouped_prediction_scores_mn['predictions'].values)
    gt_lbl = np.array(grouped_prediction_scores_mn.predictions.index.get_level_values(level='gt_lbl'))
    input_lbl = np.array(grouped_prediction_scores_mn.predictions.index.get_level_values(level='input_lbl'))
    ref_lbl = np.array(grouped_prediction_scores_mn.predictions.index.get_level_values(level='ref_lbl'))

    fpr, tpr, thresholds = metrics.roc_curve(gt_lbl, preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"overall auc_predictions min: {auc_predictions}", flush=True)
    open_preds = preds[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    open_gt_lbl = gt_lbl[((input_lbl > len(closed_classes) - 1) & (ref_lbl > len(closed_classes) - 1))]
    fpr, tpr, thresholds = metrics.roc_curve(open_gt_lbl, open_preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"openset auc_predictions min: {auc_predictions}", flush=True)
    closed_preds=preds[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]
    closed_gt_lbl=gt_lbl[((input_lbl < len(closed_classes)) & (ref_lbl < len(closed_classes)))]

    fpr, tpr, thresholds = metrics.roc_curve(closed_gt_lbl, closed_preds)
    auc_predictions = metrics.auc(fpr, tpr)
    print(f"closedset auc_predictions min: {auc_predictions}", flush=True)
    for input in range(len(classes)):
        gt_lbl_input = gt_lbl[(input_lbl == input)]
        pds = preds[(input_lbl == input)]
        fpr, tpr, thresholds = metrics.roc_curve(gt_lbl_input, pds)
        auc_predictions = metrics.auc(fpr, tpr)
        print(
            f'input: {input}, auc_predictions min: {auc_predictions}',flush=True)