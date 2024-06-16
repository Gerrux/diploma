# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import torch

import os
from imblearn.over_sampling import RandomOverSampler
import time
import numpy as np
import sklearn.model_selection
from sklearn.metrics import classification_report
import joblib
from utils.config import Config
from action_classifier.classifier import ClassifierOfflineTrain


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


def main():
    # -- setting
    cfg = Config(config_file='../configs/train_action_recogn_pipeline.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]
    classes = np.array(cfg.classes)

    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    src_features_X = get_path(cfg_stage.input.features_x)
    src_features_Y = get_path(cfg_stage.input.features_y)
    dst_model_path = cfg_stage.output.model_path

    # -- Load preprocessed data
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(src_features_X, dtype=float)  # features
    Y = np.loadtxt(src_features_Y, dtype=int)  # labels

    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.20)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # -- Balance classes in the training set
    print("\nBalancing classes in the training set ...")
    # ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    # tr_X, tr_Y = ros.fit_resample(tr_X, tr_Y)
    print("Size of balanced training data X:    ", tr_X.shape)
    print("Number of balanced training samples: ", len(tr_Y))

    # -- Train the model
    print("\nStart training model ...")
    f = [{"hidden_sizes": [2048, 1024, 1024],
          "learning_rate": 0.001,
          "num_epochs": 1200},
         ]
    for variant in f:
        output_dir_logs = f'../logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-last/'
        os.makedirs(output_dir_logs, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(output_dir_logs, 'training.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        model = ClassifierOfflineTrain(action_labels=classes,
                                       input_size=768,
                                       hidden_sizes=variant['hidden_sizes'],
                                       learning_rate=variant['learning_rate'],
                                       num_epochs=variant['num_epochs'],
                                       output_dir_logs=output_dir_logs)

        logger.info(f"Start training model with parameters: {variant}")
        model.train(tr_X, tr_Y, te_X, te_Y)

        # -- Evaluate model
        logger.info("\nStart evaluating model ...")
        evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y, logger)

        model_path = os.path.join(output_dir_logs, "action_classifier_8.pt")
        torch.save(model.clf.state_dict(), model_path)
        # -- Save model
        # print("\nSave model to " + dst_model_path)
        # os.makedirs(os.path.dirname(dst_model_path), exist_ok=True)
        # torch.save(model.clf.state_dict(), dst_model_path)
        # parent_dir = os.path.dirname(dst_model_path)
        # pca_path = os.path.join(parent_dir, "pca.pkl")
        # joblib.dump(model.pca, pca_path)


if __name__ == '__main__':
    main()
