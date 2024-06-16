from collections import deque

import numpy as np

from app.YPACm.train_classifier.action_behavior_classifier.feature_generator_vector import FeatureGenerator


class PersonMemory:
    def __init__(self, window_size):
        self.feature_generator = FeatureGenerator(window_size)
        self.scores_hist = deque(maxlen=window_size)
        self.scores = None
        self.action_stable_count = 0
        self.smooth_actions = deque(maxlen=window_size)  # Список для сглаженных действий
        self.smooth_behaviors = deque(maxlen=window_size)  # Список для сглаженных поведений
        self.last_action = None
        self.current_action = None
        self.reset()

    def reset(self):
        self.action_stable_count = 0
        self.last_action = None
        self.current_action = None
        self.feature_generator.reset()
        self.scores_hist.clear()
        self.scores = None

    def add_skeleton(self, skeleton):
        is_features_good, features = self.feature_generator.add_cur_skeleton(skeleton)
        return is_features_good, features

    def smooth_scores(self, curr_scores):
        """ Сглаживание текущих оценок предсказания
            путем взятия взвешенного среднего с предыдущими оценками.
        """
        self.scores_hist.append(curr_scores)
        weights = np.linspace(1, 0, len(self.scores_hist))
        weights /= weights.sum()
        weighted_sum = np.dot(weights, np.array(self.scores_hist))

        return weighted_sum
