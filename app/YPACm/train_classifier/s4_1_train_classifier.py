# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import torch
import os
import time
import numpy as np
import sklearn.model_selection
from sklearn.metrics import classification_report
import joblib
from utils.config import Config
from behavior_classifier.classifier import ClassifierOfflineTrain


def train_test_split(X, Y, ratio_of_test_size):
    """ Split training data by ratio """
    IS_SPLIT_BY_SKLEARN_FUNC = True

    # Use sklearn.train_test_split
    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 42
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    # Make train/test the same.
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y


def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y, logger):
    """ Evaluate accuracy and time cost """

    # Accuracy
    t0 = time.time()

    tr_accu, tr_precision, tr_recall, tr_f1, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    logger.info(f"Accuracy on training set is {tr_accu}")

    te_accu, te_precision, te_recall, te_f1, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    logger.info(f"Accuracy on testing set is {te_accu}")

    report = classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=True)
    logger.info("Accuracy report:")
    for class_name, report_data in report.items():
        if class_name == 'weighted avg':
            logger.info(f"{class_name.capitalize():<15} {report_data}")
        else:
            logger.info(f"{class_name:<15} {report_data}")

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    logger.info("Time cost for predicting one sample: "
                "{:.5f} seconds".format(average_time))


def relabel_classes(Y, normal_classes, abnormal_classes):
    """ Relabel classes to 'normal' and 'abnormal' """
    Y_new = np.copy(Y)
    for normal_class in normal_classes:
        Y_new[Y == normal_class] = 0  # 'normal' class
    for abnormal_class in abnormal_classes:
        Y_new[Y == abnormal_class] = 1  # 'abnormal' class
    return Y_new


def main():
    # -- setting
    cfg = Config(config_file='../configs/train_behavior_recogn_pipeline.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]
    # Update classes to ['normal', 'abnormal']
    classes = ['normal', 'abnormal']

    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    src_features_X = get_path(cfg_stage.input.features_x)
    src_features_Y = get_path(cfg_stage.input.features_y)
    dst_model_path = get_path(cfg_stage.output.model_path)
    output_dir_logs = f'../logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-last/'
    os.makedirs(output_dir_logs, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(output_dir_logs, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # -- Load preprocessed data
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(src_features_X, dtype=float)  # features
    Y = np.loadtxt(src_features_Y, dtype=int)  # labels

    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.2)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # -- Train the model
    print("\nStart training model ...")
    model = ClassifierOfflineTrain(num_features_from_pca=256,
                                   hidden_sizes=[2048, 1024, 1024],
                                   num_epochs=1000,
                                   learning_rate=0.001,
                                   action_labels=classes,
                                   output_dir_logs=output_dir_logs)
    model.train(tr_X, tr_Y, te_X, te_Y)

    # -- Evaluate model
    print("\nStart evaluating model ...")
    evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y, logger)

    # -- Save model
    print("\nSave model to " + output_dir_logs)
    model_path = os.path.join(output_dir_logs, "behavior_classifier_8.pt")
    torch.save(model.clf.state_dict(), model_path)
    pca_path = os.path.join(output_dir_logs, "pca.pkl")
    joblib.dump(model.pca, pca_path)
    # with open(dst_model_path, 'wb') as f:
    #     pickle.dump(model, f)


if __name__ == '__main__':
    main()
