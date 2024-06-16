# -*- coding: utf-8 -*-
"""
This script includes:

1. ClassifierOfflineTrain
    This is for offline training. The input data are the processed features.
2. class ClassifierOnlineTest(object)
    This is for online testing. The input data are the raw skeletons.
    It uses FeatureGenerator to extract features,
    and then use ClassifierOfflineTrain to recognize the action.
    Notice, this model is only for recognizing the action of one person.

TODO: Add more comments to this function.
"""
import os

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from collections import deque

from .feature_generator_vector import FeatureGenerator
from app.YPACm.train_classifier.utils.visialization_plots import Visualizer

# Import CatBoost
from catboost import CatBoostClassifier


class ClassifierOnlineTest(object):
    def __init__(self, model_path, action_labels, window_size, human_id=0, threshold=0.7, max_prev_pred=5):
        self.model = None
        self.pca = None
        self.scores_hist = None
        self.score = None
        self.model_path = model_path
        self.action_labels = action_labels
        self.threshold = threshold
        self.window_size = window_size
        self.human_id = human_id

        self.prev_predictions = deque(maxlen=max_prev_pred)
        self.prev_pred_weights = deque(maxlen=max_prev_pred)

        self.load_model()
        self.feature_generator = FeatureGenerator(window_size)

        self.reset()

    def load_model(self):
        model_path = self.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        self.model = CatBoostClassifier(task_type="GPU", devices='0')
        self.model.load_model(model_path)

    def reset(self):
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.score = None

    def predict(self, skeleton):
        is_features_good, features = self.feature_generator.add_cur_skeleton(skeleton)

        if is_features_good:
            features = features.reshape(1, -1)
            curr_scores = self.model.predict(features)
            self.score = int(curr_scores[0])

            predicted_label = [self.action_labels[self.score]]
        else:
            predicted_label = ['']
        return predicted_label

    # Улучшение техники сглаживания предсказаний
    def smooth_scores(self, curr_scores):
        """ Smooth the current prediction score
            by taking the weighted average with previous scores
        """
        self.scores_hist.append(curr_scores)
        DEQUE_MAX_SIZE = 5  # Увеличим размер буфера для сглаживания
        if len(self.scores_hist) > DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

        # Используем взвешенное среднее для сглаживания
        weights = np.linspace(1, 0, len(self.scores_hist))
        weights = weights / weights.sum()  # Нормируем веса
        weighted_sum = sum(weight * score for weight, score in zip(weights, self.scores_hist))

        return weighted_sum


class ClassifierOfflineTrain(object):
    """
    The classifer for offline training.
    The input features to this action_classifier are already
    processed by `class FeatureGenerator`.
    """

    def __init__(self, action_labels,
                 learning_rate=0.01, num_epochs=250, output_dir_logs="../logs"):
        self.action_labels = action_labels
        self.visualizer = Visualizer(output_dir=output_dir_logs)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.clf = CatBoostClassifier(iterations=num_epochs, learning_rate=learning_rate, task_type="GPU", devices='0')

    def forward(self, X):
        """ Predict the class index of the feature X """
        Y_predict = self.clf.predict(X)
        return Y_predict

    def predict_and_evaluate(self, te_X, te_Y, need_all=False):
        """ Test model on test set and obtain accuracy and other metrics """
        te_Y_predict = self.clf.predict(te_X)
        N = len(te_Y)

        n = sum(te_Y_predict == te_Y)
        accu = n / N

        precision = None
        recall = None
        f1 = None

        if need_all:
            # Calculate additional metrics
            precision = precision_score(te_Y, te_Y_predict, average='weighted')
            recall = recall_score(te_Y, te_Y_predict, average='weighted')
            f1 = f1_score(te_Y, te_Y_predict, average='weighted')

        return accu, precision, recall, f1, te_Y_predict

    def get_accuracy_per_class(self, X_test, Y_test):
        Y_pred = self.forward(X_test)
        res_confusion_matrix = confusion_matrix(Y_test, Y_pred)
        accuracy_per_class = [
            100 * res_confusion_matrix[i, i] / res_confusion_matrix.sum(axis=1)[i] if res_confusion_matrix.sum(axis=1)[
                                                                                          i] > 0 else 0
            for i in range(len(self.action_labels))]
        return accuracy_per_class

    def train(self, X, Y, X_test=None, Y_test=None):
        """ Train model. The result is saved into self.clf """
        self.clf.fit(X, Y, eval_set=(X_test, Y_test) if X_test is not None and Y_test is not None else None, plot=True)

        # Additional metrics logging (optional)
        train_acc, _, _, _, _ = self.predict_and_evaluate(X, Y)
        if X_test is not None and Y_test is not None:
            accu_test, precision_test, recall_test, f1_test, _ = self.predict_and_evaluate(X_test, Y_test,
                                                                                           need_all=True)
            print(f"Train Accuracy: {train_acc * 100:.2f}%, Test Accuracy: {accu_test * 100:.2f}%")
            print(f"Precision: {precision_test * 100:.2f}%, Recall: {recall_test * 100:.2f}%, F1: {f1_test * 100:.2f}%")

        self.visualizer.plot_training_curves(self.clf.get_evals_result()['learn']['Logloss'],
                                             self.clf.get_evals_result()['learn']['Accuracy'],
                                             self.clf.get_evals_result()['validation']['Accuracy'])

    def _choose_model(self, name):
        pass  # No need to choose a model since we are using CatBoost

    def _init_all_models(self):
        pass  # No need to initialize multiple models since we are using CatBoost

    def _predict_proba(self, X):
        """ Predict the probability of feature X belonging to each of the class Y[i] """
        Y_probs = self.clf.predict_proba(X)
        return Y_probs  # np.array with a length of len(classes)


class MultiPersonBehaviorClassifier(object):
    """
    This is a wrapper around ClassifierOnlineTest
    for recognizing actions of multiple people.
    """

    def __init__(self, name, model_path, classes, window_size=5, threshold=0.7):

        self.dict_id2clf = {}  # human id -> action_classifier of this person
        if isinstance(model_path, (list, tuple)):
            model_path = os.path.join(*model_path)
        # Define a function for creating action_classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, window_size, human_id, threshold=threshold)

    def classify(self, predictions):
        """ Classify the action type of each skeleton in dict_id2skeleton """

        dict_id2skeleton = {tracker_id: torch.flatten(keypoints.xyn).tolist() for tracker_id, keypoints in
                            zip(predictions.tracker_id, predictions.data["keypoints"])}
        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)  # check person is missed or not
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        actions = []
        for idx, (id, skeleton) in enumerate(dict_id2skeleton.items()):
            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            # actions[id] = action_classifier.predict(skeleton)  # predict label
            predictions_classifier = classifier.predict(skeleton)
            actions.append(predictions_classifier[0])
        predictions.data["behaviors"] = actions
        return predictions

    def get_classifier(self, id):
        """ Get the action_classifier based on the person id.
        Arguments:
            id {int or "min"}
        """
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]
