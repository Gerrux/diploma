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

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch import nn, optim
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from .feature_generator_vector import FeatureGenerator
from app.YPACm.train_classifier.utils.visialization_plots import Visualizer

HIDDEN_SIZES = [2048, 1024, 1024]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()

        self.hidden_layers = nn.Sequential()
        for in_size, out_size in zip([input_size] + hidden_sizes[:-1], hidden_sizes):
            self.hidden_layers.add_module(
                f"layer_{len(self.hidden_layers)}",
                nn.Sequential(
                    nn.Linear(in_size, out_size),
                    nn.BatchNorm1d(out_size),
                    nn.ELU(),
                    nn.Dropout(p=dropout_prob)
                )
            )

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


class ClassifierOnlineTest(object):
    def __init__(self, model_path, action_labels, window_size, human_id=0, threshold=0.7, max_prev_pred=5):
        self.model = None
        self.scores_hist = None
        self.scores = None
        self.model_path = model_path
        self.action_labels = action_labels
        self.threshold = threshold
        self.window_size = window_size
        self.human_id = human_id
        self.threshold = threshold
        self.prev_predictions = deque(maxlen=max_prev_pred)
        self.prev_pred_weights = deque(maxlen=max_prev_pred)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_generator = FeatureGenerator(window_size)
        self.load_model()
        self.reset()

    def load_model(self):
        self.model = MLPClassifier(
            input_size=self.feature_generator.feature_size, hidden_sizes=HIDDEN_SIZES, output_size=len(self.action_labels)
        ).eval().to(self.device)
        model_path = self.model_path

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def reset(self):
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.scores = None

    def predict(self, skeleton):
        is_features_good, features = self.feature_generator.add_cur_skeleton(skeleton)

        if is_features_good:
            features = features.reshape(1, -1)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                curr_scores = self.model(features_tensor.cuda())
            self.scores = self.smooth_scores(curr_scores.cpu().numpy()[0])
            # print(self.scores)
            if self.scores.max() < self.threshold:
                predicted_label = ['', 0]
            else:
                predicted_idx = self.scores.argmax()
                predicted_label = [self.action_labels[predicted_idx], self.scores.max()]
        else:
            predicted_label = ['', 0]

        bufer_scores = self.scores.copy()
        final_pred = predicted_label
        # # Calculate the weighted average for the final prediction
        # if len(self.prev_predictions) > 1:
        #     # Decrease the weights of previous predictions
        #     for i, weight in enumerate(self.prev_pred_weights):
        #         self.prev_pred_weights[i] *= 0.8  # Adjust this factor as needed
        #         self.prev_predictions[i] *= self.prev_pred_weights[i]
        #     # print(self.prev_pred_weights)
        #     # print(self.prev_predictions)
        #     self.scores = sum(self.prev_pred_weights) + self.scores
        #     predicted_idx = self.scores.argmax()
        #     final_pred = [self.action_labels[predicted_idx], self.scores.max()]
        # else:
        #     final_pred = predicted_label
        #
        # # Store the current prediction and weight
        # self.prev_predictions.append(bufer_scores)
        # self.prev_pred_weights.append(0.8)
        return final_pred

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

    def __init__(self, action_labels, input_size=768,
                 hidden_sizes=HIDDEN_SIZES, learning_rate=0.0001,
                 num_epochs=250, output_dir_logs="../logs"):
        self.action_labels = action_labels
        self.visualizer = Visualizer(output_dir=output_dir_logs)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self._init_all_models()

        self.clf = self._choose_model("Neural Net")

    def forward(self, X):
        """ Predict the class index of the feature X """
        Y_predict = self.clf.forward(torch.tensor(X, dtype=torch.float32).to(device))
        return Y_predict

    def predict_and_evaluate(self, te_X, te_Y, need_all=False):
        """ Test model on test set and obtain accuracy and other metrics """
        te_Y_predict = self.forward(te_X).cpu()
        N = len(te_Y)

        te_Y_predict = torch.argmax(te_Y_predict, dim=1)
        n = sum(te_Y_predict == torch.tensor(te_Y, dtype=torch.long))
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
        Y_pred = self.forward(X_test).argmax(dim=1)
        res_confusion_matrix = confusion_matrix(Y_test, Y_pred.cpu())
        accuracy_per_class = [
            100 * res_confusion_matrix[i, i] / res_confusion_matrix.sum(axis=1)[i] if res_confusion_matrix.sum(axis=1)[i] > 0 else 0
            for i in range(len(self.action_labels))]
        return accuracy_per_class

    def train(self, X, Y, X_test=None, Y_test=None):
        """ Train model. The result is saved into self.clf """
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Y = torch.tensor(Y, dtype=torch.long).to(device)
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.clf.parameters(), lr=learning_rate, weight_decay=1e-6)
        model = self.clf.to(device)
        bufer_precision_test, bufer_recall_test, bufer_f1_test = 0, 0, 0

        # Инициализация списков для хранения значений потерь и точности
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        test_accuracies_per_class = []
        # Цикл тренировки
        loop = tqdm(range(num_epochs), total=num_epochs)
        for i in loop:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == Y).sum().item()
                accuracy = correct / len(Y)
                train_losses.append(loss.item())
                train_accuracies.append(accuracy)

                if X_test is not None and Y_test is not None:
                    if i % 10 == 0:
                        accu_test, bufer_precision_test, bufer_recall_test, bufer_f1_test, _ = self.predict_and_evaluate(X_test, Y_test, need_all=True)
                        test_accuracies.append(accu_test)
                        test_accuracies_per_class.append(self.get_accuracy_per_class(X_test, Y_test))
                    else:
                        accu_test, _, _, _, _ = self.predict_and_evaluate(X_test, Y_test, need_all=False)
                        test_accuracies.append(accu_test)
            loop.set_description('Epoch: ')
            loop.set_postfix(info=f"Loss: {loss.item():.4f}, "
                                  f"Accuracy: train={accuracy * 100:.2f}%, "
                                  f"test={accu_test * 100:.2f}%, "
                                  f"Precision: test={bufer_precision_test * 100:.2f}%, "
                                  f"Recall: test={bufer_recall_test * 100:.2f}%, "
                                  f"F1: test={bufer_f1_test * 100:.2f}%")
        loop.close()

        self.visualizer.plot_training_curves(train_losses, train_accuracies, test_accuracies)
        epochs = list(range(0, num_epochs + 1, 10))[:len(test_accuracies_per_class)]
        self.visualizer.plot_accuracy_per_class(epochs, test_accuracies_per_class, self.action_labels)
        self.visualizer.plot_conf_matrix(Y_test, self.forward(X_test).cpu().argmax(dim=1), self.action_labels)
        self.clf = model

    def _choose_model(self, name):
        self.model_name = name
        idx = self.names.index(name)
        return self.classifiers[idx]

    def _init_all_models(self):
        self.names = ["Neural Net"]
        self.model_name = None
        self.classifiers = [
            MLPClassifier(input_size=self.input_size,
                          hidden_sizes=self.hidden_sizes,
                          output_size=len(self.action_labels))]

    def _predict_proba(self, X):
        """ Predict the probability of feature X belonging to each of the class Y[i] """
        Y_probs = self.clf.predict_proba(X)
        return Y_probs  # np.array with a length of len(classes)


class MultiPersonActionClassifier(object):
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
            # print(predictions_classifier)
            actions.append(predictions_classifier[0])
        predictions.data["actions"] = actions
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
