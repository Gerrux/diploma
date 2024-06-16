import os
import time
import json
import cv2
import torch
from tqdm import tqdm
from tabulate import tabulate

from utils.config import Config
from utils.skeletons_io import ReadValidImagesAndActionTypesByTxt
from ultralytics import YOLO


def main():
    t0 = time.time()
    # Settings
    cfg = Config(config_file='../configs/train_behavior_recogn_pipeline.yaml')
    cfg.merge_from_file('../configs/infer_yolov8pose_dnn.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]
    img_format = cfg.img_format

    # IO folders
    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    src_imgs_folder = get_path(cfg_stage.input.imgs_folder)
    src_valid_imgs = get_path(cfg_stage.input.valid_imgs)
    dst_skeletons_folder = get_path(cfg_stage.output.skeletons_folder)
    dst_imgs_folder = get_path(cfg_stage.output.imgs_folder)
    dst_imgs_info_txt = get_path(cfg_stage.output.imgs_info_txt)

    # initiate pose estimator
    pose_config = cfg.POSE
    pose_estimator = YOLO("."+get_path(pose_config['model_path']))

    # Init output path
    print(f"[INFO] Creating output folder -> {os.path.dirname(dst_skeletons_folder)}")
    os.makedirs(dst_imgs_folder, exist_ok=True)
    os.makedirs(dst_skeletons_folder, exist_ok=True)
    os.makedirs(os.path.dirname(dst_imgs_info_txt), exist_ok=True)

    # train val images reader
    images_loader = ReadValidImagesAndActionTypesByTxt(src_imgs_folder,
                                                       src_valid_imgs,
                                                       img_format)
    images_loader.save_images_info(dst_imgs_info_txt)
    print(f'[INFO] Total Images -> {len(images_loader)}')

    # Read images and process
    loop = tqdm(range(len(images_loader)), total=len(images_loader))
    for i in loop:
        img_bgr, label, img_info = images_loader.read_image()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # predict yolov8pose skeleton and save to file as openpose format
        predictions = pose_estimator.predict(img_rgb, verbose=False, conf=0.5)

        if len(predictions) == 0: continue
        predictions = predictions[0].keypoints.xyn

        # save predicted image
        save_name = img_format.format(i)
        # save skeletons in text file
        skeleton_txt = os.path.join(dst_skeletons_folder, save_name[:-4] + '.txt')
        save_data = [img_info + torch.flatten(pred).tolist() for pred in predictions]
        with open(skeleton_txt, 'w') as f:
            json.dump(save_data, f)

        # update progress bar descriptions
        loop.set_description(f'action -> {label}')
        loop.set_postfix(num_of_person=len(predictions))

    loop.close()
    t1 = time.gmtime(time.time() - t0)
    total_time = time.strftime("%H:%M:%S", t1)

    print('Total Extraction Time', total_time)
    print(tabulate([list(images_loader.labels_info.values())],
                   list(images_loader.labels_info.keys()), 'grid'))


if __name__ == '__main__':
    main()
