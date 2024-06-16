import math
from collections import deque
import numpy as np
from scipy.signal import medfilt


class FeatureGenerator:
    # Определение ключевых точек COCO и их индексов
    POSE_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle"
    ]
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # Определение углов и расстояний для вычисления
    ANGLES = [
        (LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, math.pi),  # Угол левого локтя
        (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST, math.pi),  # Угол правого локтя
        (LEFT_KNEE, LEFT_HIP, LEFT_ANKLE, math.pi),  # Угол левого колена
        (RIGHT_KNEE, RIGHT_HIP, RIGHT_ANKLE, math.pi),  # Угол правого колена
        (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, math.pi),  # Угол плеч
        (RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_ELBOW, math.pi),  # Угол плеч
        (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, math.pi/2),  # Угол бедра
        (RIGHT_HIP, LEFT_HIP, RIGHT_KNEE, math.pi / 2),  # Угол бедра
        # (LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER, math.pi),  # Угол левого запястья
        # (RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, math.pi),  # Угол правого запястья
    ]

    DISTANCES = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),  # Расстояние между плечами
        (LEFT_HIP, RIGHT_HIP),  # Расстояние между бедрами
        (LEFT_SHOULDER, LEFT_HIP),  # Расстояние между левым плечом и бедром
        (RIGHT_SHOULDER, RIGHT_HIP),  # Расстояние между правым плечом и бедром
        (LEFT_ELBOW, LEFT_KNEE),  # Расстояние между левым локтем и коленом
        (RIGHT_ELBOW, RIGHT_KNEE),  # Расстояние между правым локтем и коленом
        (LEFT_SHOULDER, RIGHT_HIP),  # Расстояние между левым плечом и правым бедром
        (RIGHT_SHOULDER, LEFT_HIP),  # Расстояние между правым плечом и левым бедром
        (LEFT_WRIST, LEFT_ANKLE),  # Расстояние между левым запястьем и лодыжкой
        (RIGHT_WRIST, RIGHT_ANKLE),  # Расстояние между правым запястьем и лодыжкой
        (LEFT_WRIST, RIGHT_ANKLE),  # Расстояние между левым запястьем и правой лодыжкой
        (RIGHT_WRIST, LEFT_ANKLE),  # Расстояние между правым запястьем и левой лодыжкой
    ]

    LIMB_LENGTHS = [
        (LEFT_SHOULDER, LEFT_ELBOW),  # Длина левой верхней части руки
        (RIGHT_SHOULDER, RIGHT_ELBOW),  # Длина правой верхней части руки
        (LEFT_ELBOW, LEFT_WRIST),  # Длина левой нижней части руки
        (RIGHT_ELBOW, RIGHT_WRIST),  # Длина правой нижней части руки
        (LEFT_HIP, LEFT_KNEE),  # Длина левой верхней части ноги
        (RIGHT_HIP, RIGHT_KNEE),  # Длина правой верхней части ноги
        (LEFT_KNEE, LEFT_ANKLE),  # Длина левой нижней части ноги
        (RIGHT_KNEE, RIGHT_ANKLE),  # Длина правой нижней части ноги
    ]

    LEG_AND_ARM_JOINTS = [
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
    ]

    ADDITIONAL_ANGLES = [
        (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, math.pi),  # Угол между плечом, бедром и коленом слева
        (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, math.pi),  # Угол между плечом, бедром и коленом справа
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, math.pi),  # Угол между бедром, коленом и лодыжкой слева
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, math.pi),  # Угол между бедром, коленом и лодыжкой справа
    ]

    ADDITIONAL_DISTANCES = [
        (LEFT_ELBOW, RIGHT_ELBOW),  # Расстояние между локтями
        (LEFT_WRIST, RIGHT_WRIST),  # Расстояние между запястьями
        (LEFT_KNEE, RIGHT_KNEE),  # Расстояние между коленями
        (LEFT_ANKLE, RIGHT_ANKLE),  # Расстояние между лодыжками
    ]

    def __init__(self, window_size, fps=30, smoothed_filter='Mean Average', is_adding_noise=False):
        """
        Инициализация класса FeatureGenerator.
        :param window_size: Количество кадров для расчета признаков.
        :param fps: Частота кадров видео данных.
        :param is_adding_noise: Добавлять ли шум в данные скелета.
        """
        self._window_size = window_size
        self._fps = fps
        self._frame_time_diff = 1 / fps
        self._is_adding_noise = is_adding_noise
        self._noise_intensity = 0.01
        self.prev_skeleton = None
        self.smoothed_filter = smoothed_filter
        self.reset()
        self.feature_size = (3 * (len(self.ANGLES) + len(self.ADDITIONAL_ANGLES)) + len(self.DISTANCES)
                             + len(self.ADDITIONAL_DISTANCES) + len(self.LIMB_LENGTHS)
                             + 3 * len(self.LEG_AND_ARM_JOINTS)) * window_size
    def reset(self):
        """Сброс внутреннего состояния класса FeatureGenerator."""
        self._x_deque = deque()
        self.prev_skeleton = None
        self._pre_x = None

    def add_cur_skeleton(self, skeleton):
        """
        Добавить текущие данные скелета во внутреннюю очередь и, если возможно, рассчитать признаки.
        :param skeleton: Данные скелета в формате COCO.
        :return: Кортеж, указывающий на успешность вычисления признаков и сами вычисленные признаки.
        """
        if skeleton is None or len(skeleton) == 0:
            self.reset()
            return False, None

        self.prev_skeleton = skeleton

        x = self._process_skeleton(skeleton)

        if self._is_adding_noise:
            x = self._add_noises(x, self._noise_intensity)
        x = np.array(x)
        self._x_deque.append(x)
        self._maintain_deque_size()
        self._pre_x = x.copy()

        if len(self._x_deque) < self._window_size:
            self._x_deque.extendleft([np.zeros_like(self._pre_x)] * (self._window_size - len(self._x_deque)))

        features = self._extract_features(self._x_deque)
        # Применение фильтрации для сглаживания признаков
        if self.smoothed_filter == 'Mean Average':
            features = self.apply_moving_average_filter(features, window_size=3)
        elif self.smoothed_filter == 'Median':
            features = self.apply_median_filter(features, window_size=3)
        return True, features

    def _maintain_deque_size(self):
        """Поддерживать размер внутренней очереди, удаляя самый старый кадр, если необходимо."""
        if len(self._x_deque) > self._window_size:
            self._x_deque.popleft()

    def _process_skeleton(self, skeleton_data):
        """
        Обработка данных скелета для сохранения только допустимых суставов.
        :param skeleton_data: Данные скелета в формате COCO.
        :return: Numpy массив, содержащий только допустимые суставы.
        """
        num_joints = len(self.POSE_KEYPOINTS)

        if len(skeleton_data) != 2 * num_joints:
            raise ValueError(f"Неверный размер данных скелета. Ожидалось: {2 * num_joints}, получено: {len(skeleton_data)}")

        # Извлечение только координат суставов из входных данных
        skeleton = skeleton_data[-2 * num_joints:]
        skeleton = np.reshape(skeleton, (num_joints, 2))

        # Расчет средней точки между плечами как эталонной точки для нормализации
        # midpoint = (skeleton[self.LEFT_SHOULDER] + skeleton[self.RIGHT_SHOULDER]) / 2
        # skeleton = np.concatenate([skeleton, [midpoint]])

        return skeleton

    def _add_noises(self, data, noise_intensity):
        """
        Добавление шума в данные скелета.
        :param data: Данные скелета.
        :param noise_intensity: Интенсивность шума.
        :return: Данные скелета с добавленным шумом.
        """
        noise = np.random.normal(scale=noise_intensity, size=data.shape)
        return data + noise

    def _calculate_angle_vectorized(self, v1, v2, v3):
        """
        Calculate the angle between three joints for a vector of joints.
        :param v1: The coordinates of the first joint as a NumPy array.
        :param v2: The coordinates of the second joint as a NumPy array.
        :param v3: The coordinates of the third joint as a NumPy array.
        :return: The calculated angle in radians as a NumPy array.
        """
        eps = 1e-8
        v1 = v1 - v2
        v2 = v3 - v2
        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + eps
        dot_product = np.einsum('ij,ij->i', v1, v2)
        angle = np.arccos(np.clip(dot_product / norms, -1, 1))
        return angle

    def _extract_features(self, x_deque):
        num_angles = len(self.ANGLES) + len(self.ADDITIONAL_ANGLES)
        num_distances = len(self.DISTANCES) + len(self.ADDITIONAL_DISTANCES)
        num_speeds = len(self.LEG_AND_ARM_JOINTS)
        num_limb_lengths = len(self.LIMB_LENGTHS)
        num_angle_changes = len(self.ANGLES) + len(self.ADDITIONAL_ANGLES)
        num_angular_velocities = len(self.ANGLES) + len(self.ADDITIONAL_ANGLES)
        num_linear_velocities = len(self.LEG_AND_ARM_JOINTS)
        num_accelerations = len(self.LEG_AND_ARM_JOINTS)
        total_features = (num_angles + num_distances + num_speeds + num_limb_lengths +
                          num_angle_changes + num_angular_velocities + num_linear_velocities + num_accelerations)
        features = np.zeros((total_features, self._window_size))

        # Calculate height_body once before the loop
        shoulder_midpoint = np.linalg.norm(x_deque[0][self.RIGHT_SHOULDER] - x_deque[0][self.LEFT_SHOULDER])
        hip_midpoint = np.linalg.norm(x_deque[0][self.RIGHT_HIP] - x_deque[0][self.LEFT_HIP])
        height_body = abs(hip_midpoint - shoulder_midpoint)

        if height_body == 0:
            height_body = 1
        # Interpolate missing values for the entire x_deque
        x_deque = np.array([self._interpolate_missing_values(x) for x in x_deque])

        # Calculate the angles between joints using NumPy
        for j, (joint1, joint2, joint3, base_angle) in enumerate(self.ANGLES + self.ADDITIONAL_ANGLES):
            angle = self._calculate_angle_vectorized(x_deque[:, joint1], x_deque[:, joint2], x_deque[:, joint3])
            features[j, :] = angle / base_angle

        # Calculate the distances between joints using NumPy
        for j, distance_joints in enumerate(self.DISTANCES + self.ADDITIONAL_DISTANCES):
            distance = np.linalg.norm(x_deque[:, distance_joints[0]] - x_deque[:, distance_joints[1]], axis=1)
            features[num_angles + j, :] = distance * height_body  # Масштабирование по высоте тела

        # Calculate the speeds of joints
        for j, joint in enumerate(self.LEG_AND_ARM_JOINTS):
            speeds = np.zeros(self._window_size)
            speeds[1:] = np.linalg.norm(x_deque[1:, joint] - x_deque[:-1, joint], axis=1) / self._frame_time_diff
            normalized_speeds = speeds * height_body  # Масштабирование по высоте тела
            features[num_angles + num_distances + j, :] = normalized_speeds

        # Calculate limb lengths using NumPy
        for j, limb_joints in enumerate(self.LIMB_LENGTHS):
            limb_length = np.linalg.norm(x_deque[:, limb_joints[0]] - x_deque[:, limb_joints[1]], axis=1)
            features[num_angles + num_distances + num_speeds + j, :] = limb_length * height_body  # Масштабирование по высоте тела

        # Calculate angle changes using NumPy
        for j, (joint1, joint2, joint3, base_angle) in enumerate(self.ANGLES + self.ADDITIONAL_ANGLES):
            prev_angle = self._calculate_angle_vectorized(x_deque[:-1, joint1], x_deque[:-1, joint2],
                                                          x_deque[:-1, joint3])
            curr_angle = self._calculate_angle_vectorized(x_deque[1:, joint1], x_deque[1:, joint2], x_deque[1:, joint3])
            angle_change = np.abs(curr_angle - prev_angle) / base_angle
            features[num_angles + num_distances + num_speeds + num_limb_lengths + j, 1:] = angle_change

        # Calculate the angular velocities using NumPy
        for j, (joint1, joint2, joint3, base_angle) in enumerate(self.ANGLES + self.ADDITIONAL_ANGLES):
            prev_angle = self._calculate_angle_vectorized(x_deque[:-1, joint1], x_deque[:-1, joint2],
                                                          x_deque[:-1, joint3])
            curr_angle = self._calculate_angle_vectorized(x_deque[1:, joint1], x_deque[1:, joint2],
                                                          x_deque[1:, joint3])
            angular_velocity = np.abs(curr_angle - prev_angle) / base_angle / self._frame_time_diff  # Угловая скорость
            features[
            num_angles + num_distances + num_speeds + num_limb_lengths + num_angle_changes + j, 1:] = angular_velocity

        # Calculate the normalized linear velocities using NumPy
        for j, joint in enumerate(self.LEG_AND_ARM_JOINTS):
            speeds = np.zeros(self._window_size)
            speeds[1:] = np.linalg.norm(x_deque[1:, joint] - x_deque[:-1, joint], axis=1) / self._frame_time_diff
            normalized_speeds = speeds * height_body  # Нормализованная линейная скорость
            features[
            num_angles + num_distances + num_speeds + num_limb_lengths + num_angle_changes + num_angular_velocities + j,
            :] = normalized_speeds

        # Calculate the accelerations using NumPy
        for j, joint in enumerate(self.LEG_AND_ARM_JOINTS):
            if self._window_size > 2:
                prev_speed = np.linalg.norm(x_deque[1:-1, joint] - x_deque[:-2, joint], axis=1) / self._frame_time_diff
                curr_speed = np.linalg.norm(x_deque[2:, joint] - x_deque[1:-1, joint], axis=1) / self._frame_time_diff
                acceleration = np.abs(curr_speed - prev_speed) / self._frame_time_diff
                normalized_acceleration = acceleration * height_body  # Нормализованное ускорение
                features[
                num_angles + num_distances + num_speeds + num_limb_lengths + num_angle_changes +
                num_angular_velocities + num_linear_velocities + j, 1:-1] = normalized_acceleration

        self._pre_x = x_deque[-1]  # Avoid copying the array if not necessary

        return features.flatten()

    @staticmethod
    def _interpolate_missing_values(x):
        """
        Interpolate missing values in the skeleton data.
        :param x: The skeleton data as a numpy array.
        :return: The skeleton data with interpolated missing values.
        """
        for i in range(len(x)):
            if np.any(np.isnan(x[i])):
                if i > 0 and not np.any(np.isnan(x[i - 1])):
                    x[i] = x[i - 1]
                elif i < len(x) - 1 and not np.any(np.isnan(x[i + 1])):
                    x[i] = x[i + 1]
                else:
                    x[i] = 0
        return x

    def _calculate_speed(self, x_deque, index, joint):
        """
        Calculate the speed of a joint.
        :param x_deque: The internal deque containing the skeleton data.
        :param index: The index of the current frame.
        :param joint: The index of the joint to calculate the speed for.
        :return: The speed of the joint.
        """
        if index == 0:
            return 0.0
        prev_x = x_deque[index - 1]
        curr_x = x_deque[index]
        distance = np.linalg.norm(curr_x[joint] - prev_x[joint])
        speed = distance / self._frame_time_diff
        return speed

    def apply_moving_average_filter(self, features, window_size):
        """
        Apply a moving average filter to the features.
        :param features: The extracted features as a numpy array.
        :param window_size: The size of the moving window.
        :return: The smoothed features.
        """
        smoothed_features = np.convolve(features, np.ones(window_size) / window_size, mode='valid')
        padding_size = len(features) - len(smoothed_features)
        if padding_size > 0:
            smoothed_features = np.pad(smoothed_features, (padding_size, 0), mode='edge')
        return smoothed_features

    def apply_median_filter(self, features, window_size):
        """
        Apply a median filter to the features.
        :param features: The extracted features as a numpy array.
        :param window_size: The size of the moving window.
        :return: The smoothed features.
        """
        smoothed_features = medfilt(features, kernel_size=window_size)
        return smoothed_features


if __name__ == '__main__':
    # Пример использования класса FeatureGenerator
    fg = FeatureGenerator(window_size=8, fps=30, is_adding_noise=False)
    print(fg.feature_size)
    # Здесь вы можете добавить тестовые данные для skeleton_data
    skeleton_data = [
        0.5, 0.5,  # nose
        0.45, 0.4,  # left_eye
        0.55, 0.4,  # right_eye
        0.4, 0.5,  # left_ear
        0.6, 0.5,  # right_ear
        0.3, 0.6,  # left_shoulder
        0.7, 0.6,  # right_shoulder
        0.25, 0.7,  # left_elbow
        0.75, 0.7,  # right_elbow
        0.2, 0.8,  # left_wrist
        0.8, 0.8,  # right_wrist
        0.3, 0.8,  # left_hip
        0.7, 0.8,  # right_hip
        0.25, 0.9,  # left_knee
        0.75, 0.9,  # right_knee
        0.2, 1.0,  # left_ankle
        0.8, 1.0  # right_ankle
    ]  # Добавьте ваши тестовые данные здесь
    success, features = fg.add_cur_skeleton(skeleton_data)
    if success:
        print("Features: ", features.shape)
    else:
        print("Failed to extract features.")
