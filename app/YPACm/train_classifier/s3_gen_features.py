import os
import numpy as np
from tqdm import tqdm

from utils.config import Config
from utils.skeletons_io import load_skeleton_data
from action_classifier.feature_generator_vector import FeatureGenerator

IS_ADDING_NOISE = False


def process_features(X0, window_size=5, is_adding_noise=False):
    feature_generator = FeatureGenerator(window_size, fps=30, is_adding_noise=is_adding_noise)
    X = []
    for i in tqdm(range(len(X0))):
        status_features, (features, smoothed_features_ma, smoothed_features_median) = feature_generator.add_cur_skeleton(X0[i])
        if status_features:
            X.append(smoothed_features_ma)
    return np.array(X)


def main():
    cfg = Config(config_file='../configs/train_action_recogn_pipeline.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]
    classes = np.array(cfg.classes)
    window_size = cfg.window_size
    is_adding_noise = IS_ADDING_NOISE  # Add this line if your new FeatureGenerator requires this parameter

    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    src_skeletons_txt = get_path(cfg_stage.input.skeletons_txt)
    dst_features_X = get_path(cfg_stage.output.features_x)
    dst_features_Y = get_path(cfg_stage.output.features_y)

    X0, Y0, video_indices = load_skeleton_data(src_skeletons_txt, classes)
    print(f"X0 {len(X0)}, Y0 {len(Y0)}")
    print(f"video indices {len(video_indices)}")

    print("\nExtracting time-serials features ...")
    X = process_features(X0, window_size,
                         is_adding_noise)  # Add this parameter if your new FeatureGenerator requires it
    print(f"X.shape = {X.shape}, len(Y0) = {len(Y0)}")
    print("\nWriting features and labels to disk ...")
    os.makedirs(os.path.dirname(dst_features_X), exist_ok=True)
    os.makedirs(os.path.dirname(dst_features_Y), exist_ok=True)

    np.savetxt(dst_features_X, X, fmt="%.5f")
    print("Save features to: " + dst_features_X)

    np.savetxt(dst_features_Y, Y0, fmt="%i")
    print("Save labels to: " + dst_features_Y)


if __name__ == "__main__":
    main()
