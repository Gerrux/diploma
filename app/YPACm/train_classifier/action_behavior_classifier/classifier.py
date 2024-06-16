import os
import time

from catboost import CatBoostClassifier
from torch import nn
import torch
from .person_memory import PersonMemory

HIDDEN_SIZES = [2048, 1024, 1024]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x


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


class MultiPersonClassifier:
    def __init__(self, action_classifier_path, action_labels, behavior_classifier_path, behavior_labels,
                 input_size_action_classifier=768, window_size=8, threshold=0.7, stable_threshold=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_classifier_path = get_path(action_classifier_path)
        self.input_size_action_classifier = input_size_action_classifier
        self.action_labels = action_labels
        self.threshold = threshold
        self.stable_threshold = stable_threshold
        self.action_classifier = self._load_action_classifier()

        self.behavior_classifier_path = get_path(behavior_classifier_path)
        self.behavior_labels = behavior_labels
        self.behavior_classifier = self._load_behavior_classifier()

        self.window_size = window_size
        self.person_memories = {}

    def _load_action_classifier(self):
        model = MLPClassifier(
            input_size=self.input_size_action_classifier, hidden_sizes=HIDDEN_SIZES, output_size=len(self.action_labels)
        ).eval().to(self.device)
        if not os.path.isfile(self.action_classifier_path):
            raise FileNotFoundError(f"Model file not found at path: {self.action_classifier_path}")
        model.load_state_dict(torch.load(self.action_classifier_path))
        model.eval()
        return model

    def _load_behavior_classifier(self):
        model_path = self.behavior_classifier_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        model = CatBoostClassifier(task_type="GPU", devices="0")
        model = model.load_model(model_path)
        return model

    def action_predict(self, features_batch):
        features_tensor = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = self.action_classifier(features_tensor)
        return scores.cpu().numpy()

    def behavior_predict(self, features_batch):
        features_batch = features_batch.reshape(len(features_batch), -1)
        predictions = self.behavior_classifier.predict(features_batch)
        return predictions

    def postprocess_predictions(self, person_memory, action, behavior):
        # Используемая здесь логика для постобработки
        person_memory.smooth_actions.append(action[0])  # добавляем действие в список smooth_actions
        if len(person_memory.smooth_actions) > self.window_size:
            person_memory.smooth_actions.pop(0)  # если превышен размер окна, удаляем старое действие
        smoothed_action = max(set(person_memory.smooth_actions), key=person_memory.smooth_actions.count)

        if person_memory.last_action != smoothed_action:
            person_memory.action_stable_count = 0
            person_memory.last_action = smoothed_action
        else:
            person_memory.action_stable_count += 1

        if person_memory.action_stable_count >= self.stable_threshold:
            final_action = smoothed_action
        else:
            final_action = person_memory.current_action

        person_memory.current_action = final_action
        person_memory.smooth_behaviors.append(behavior)  # добавляем поведение в список smooth_behaviors
        if len(person_memory.smooth_behaviors) > self.window_size:
            person_memory.smooth_behaviors.pop(0)  # если превышен размер окна, удаляем старое поведение
        final_behavior = max(set(person_memory.smooth_behaviors), key=person_memory.smooth_behaviors.count)

        return final_action, final_behavior

    def predict(self, predictions):
        dict_id2skeleton = {
            tracker_id: torch.flatten(keypoints.xyn).tolist()
            for tracker_id, keypoints in zip(predictions.tracker_id, predictions.data["keypoints"])
        }

        # Clear people not in view
        old_ids = set(self.person_memories)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human_id in humans_not_in_view:
            del self.person_memories[human_id]

        batch_skeletons = []
        batch_ids = []

        for human_id, skeleton in dict_id2skeleton.items():
            if human_id not in self.person_memories:
                self.person_memories[human_id] = PersonMemory(self.window_size)

            person_memory = self.person_memories[human_id]
            is_features_good, features = person_memory.add_skeleton(skeleton)
            if is_features_good:
                batch_skeletons.append(torch.tensor(features.reshape(-1), dtype=torch.float32))  # Преобразуем в тензор
                batch_ids.append(human_id)

        results = {
            'actions': [],
            'behaviors': []
        }

        if batch_skeletons:
            batch_skeletons = torch.stack(batch_skeletons)
            action_scores_batch = self.action_predict(batch_skeletons)
            behavior_predictions_batch = self.behavior_predict(batch_skeletons.numpy())  # Преобразование в numpy

            for i, human_id in enumerate(batch_ids):
                person_memory = self.person_memories[human_id]
                person_memory.scores = person_memory.smooth_scores(action_scores_batch[i])

                if person_memory.scores.max() < self.threshold:
                    action = ['', 0]
                else:
                    predicted_idx = person_memory.scores.argmax()
                    action = [self.action_labels[predicted_idx], person_memory.scores.max()]

                behavior = self.behavior_labels[behavior_predictions_batch[i]]

                # Применение постобработки
                final_action, final_behavior = self.postprocess_predictions(person_memory, action, behavior)

                results['actions'].append(final_action)
                results['behaviors'].append(final_behavior)

            predictions.data['actions'] = results['actions']
            predictions.data['behaviors'] = results['behaviors']
        torch.cuda.empty_cache()
        return predictions

    def reset(self, human_id=None):
        if human_id is not None:
            if human_id in self.person_memories:
                self.person_memories[human_id].reset()
        else:
            for memory in self.person_memories.values():
                memory.reset()
