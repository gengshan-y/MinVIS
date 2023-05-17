# modified by Gengshan Yang
# python preprocess/third_party/MinVIS/extract_tracks.py cat-pikachu-0-0000 database/processed/ quad

# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import glob
import multiprocessing as mp
import os
import pdb
import cv2
import subprocess

# fmt: off
import sys
sys.path.insert(
    0,
    "%s/"
    % os.path.join(os.path.dirname(__file__)),
)
sys.path.insert(
    0,
    "%s/demo_video/"
    % os.path.join(os.path.dirname(__file__)),
)
# fmt: on

import time

import numpy as np

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import add_minvis_config
from predictor import VisualizationDemo


def setup_cfg():
    # load config from file and command-line arguments
    #config_file = "preprocess/third_party/MinVIS/configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml"
    config_file = "preprocess/third_party/MinVIS/configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml"
    opts = [
        "MODEL.WEIGHTS",
        "preprocess/third_party/MinVIS/demo_video/minvis_ytvis21_swin_large.pth",
        #"preprocess/third_party/MinVIS/demo_video/minvis_ovis_swin_large.pth",
    ]
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def extract_tracks(seqname, outdir, obj_class):
    if obj_class == "human":
        is_human = 1  
    elif obj_class == "quad":
        is_human = 0
    else:
        raise NotImplementedError

    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    cfg = setup_cfg()

    demo = VisualizationDemo(cfg)

    video_root = "%s/JPEGImages/Full-Resolution/%s" % (outdir, seqname)
    output_root = video_root.replace("JPEGImages", "Annotations")

    os.makedirs(output_root, exist_ok=True)

    frames_path = video_root
    frames_path = glob.glob(os.path.expanduser(os.path.join(frames_path, "*.jpg")))
    frames_path.sort()

    vid_frames = []
    for path in frames_path:
        img = read_image(path, format="BGR")
        vid_frames.append(img)

    start_time = time.time()
    with autocast():
        predictions, visualized_output = demo.run_on_video(vid_frames)
    print(
        "detected {} instances per frame in {:.2f}s".format(
            len(predictions["pred_scores"]), time.time() - start_time
        )
    )

    # save frames
    label = predictions["pred_labels"]
    pred_is_human = np.asarray(label)==25 # in occvis, human=0; in ytvis, human=25
    # class label for youtubevis/ytvis
    # person: 25
    # cat: 5
    # dog: 8
    invalid_idx = pred_is_human!=is_human

    # best hypothesis
    scores = np.asarray(predictions["pred_scores"])
    scores[..., invalid_idx] = 0
    if scores.sum() == 0:
            print("Warning: no valid mask")
    best_idx = scores.sum(0).argmax(0)
    for path, _vis_output, mask, score in zip(
        frames_path,
        visualized_output,
        predictions["pred_masks"].permute(1, 0, 2, 3),  # T, K, H, W
        predictions["pred_scores"],
    ):
        # # assuming single object
        # score = np.asarray(score)
        # score[invalid_idx] = 0
        # if score.sum() == 0:
        #     print("Warning: no valid mask")

        # mask = mask.numpy() * (score[:, None, None] > 0.9)
        # mask = mask.sum(0).astype(np.int8) * 127

        # best hypothesis
        mask = mask[best_idx].numpy().astype(np.int8) * 127

        if mask.sum() == 0:
            mask[:] = -1

        out_filename = os.path.join(output_root, os.path.basename(path))
        cv2.imwrite(out_filename, mask)
        np.save(out_filename.replace(".jpg", ".npy"), mask)

    # save video
    cmd = f'cat {output_root}/*.jpg | ffmpeg -y -f image2pipe -i - -vf "scale=-1:360" -loglevel panic {output_root}/vis.mp4'
    subprocess.run(cmd, shell=True, check=True)

    print("minvis done: ", seqname)


if __name__ == "__main__":
    seqname = sys.argv[1]
    outdir = sys.argv[2]
    obj_class = sys.argv[3]
    extract_tracks(seqname, outdir, obj_class)
