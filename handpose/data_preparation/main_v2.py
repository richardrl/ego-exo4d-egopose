import copy
import json
import os

import cv2
import numpy as np
from dataloader import ego_pose_anno_loader
from PIL import Image
from projectaria_tools.core import calibration
from scripts.download import find_annotated_takes
from tqdm import tqdm
from utils.config import create_arg_parse
from utils.reader import PyAvReader
from utils.utils import extract_aria_calib_to_json, get_ego_aria_cam_name
import os, glob, cv2, json
import multiprocessing
import pyvrs


def sort_f(filepath):
    return int(os.path.basename(filepath).split(".jpg")[0])

def undistort_take_single(input_tuple):
    take_name, args, dist_img_root, undist_img_root = input_tuple
    # Get aria calibration model and pinhole camera model
    curr_aria_calib_json_path = os.path.join(
        args.gt_output_dir, "aria_calib_json", f"{take_name}.json"
    )
    if not os.path.exists(curr_aria_calib_json_path):
        print(f"No Aria calib json for {take_name}. Skipped.")
        return None
        # continue

    # modify it to have the right focal length
    aria_json = json.load(open(curr_aria_calib_json_path))

    # 1404 original image
    # 448 new image
    # f
    if args.resolution == 1404:
        pass
    elif args.resolution == 448:
        aria_json['CameraCalibrations'][4]['Projection']['Params'][0] = \
        aria_json['CameraCalibrations'][4]['Projection']['Params'][0] * (1 / 3.13392857)

        # cx, cy
        aria_json['CameraCalibrations'][4]['Projection']['Params'][1] = 224
        aria_json['CameraCalibrations'][4]['Projection']['Params'][2] = 224

    aria_rgb_calib = calibration.device_calibration_from_json_string(
        json.dumps(aria_json)
    ).get_camera_calib("camera-rgb")

    pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
    # Input and output directory
    curr_dist_img_dir = os.path.join(dist_img_root, take_name)
    curr_undist_img_dir = os.path.join(undist_img_root, take_name)
    os.makedirs(curr_undist_img_dir, exist_ok=True)

    sorted_jpgs = sorted(glob.glob(os.path.join(dist_img_root, take_name, "[0-9]*.jpg")), key=sort_f)
    assert len(sorted_jpgs) > 0, "Are you sure you ran extract_aria_img?"

    for jpg_idx, curr_dist_img_path in tqdm(enumerate(sorted_jpgs)):
        f_idx = jpg_idx + 1
        curr_undist_img_path = os.path.join(
            curr_undist_img_dir, f"{f_idx:06d}.jpg"
        )
        # Avoid repetitive generation by checking file existence
        if not os.path.exists(curr_undist_img_path):
            # Load in distorted images
            curr_dist_img_path = os.path.join(
                curr_dist_img_dir, f"{f_idx:06d}.jpg"
            )
            assert os.path.exists(
                curr_dist_img_path
            ), f"No distorted images found at {curr_dist_img_path}. Please extract images with steps=raw_images first."
            curr_dist_image = np.array(Image.open(curr_dist_img_path))
            curr_dist_image = (
                cv2.rotate(curr_dist_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if args.portrait_view
                else curr_dist_image
            )

            # Undistortion
            undistorted_image = calibration.distort_by_calibration(
                curr_dist_image, pinhole, aria_rgb_calib
            )
            undistorted_image = (
                cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)
                if args.portrait_view
                else undistorted_image
            )
            # print(f"Saving undistorted image: {curr_undist_img_path}")
            # Save undistorted image
            assert cv2.imwrite(
                curr_undist_img_path, undistorted_image[:, :, ::-1]
            ), curr_undist_img_path


def undistort_aria_img(args):
    # Load all takes metadata
    # Input and output root path
    img_view_prefix = "image_portrait_view" if args.portrait_view else "image"
    dist_img_root = os.path.join(
        args.gt_output_dir, img_view_prefix, "distorted", args.data_name
    )
    undist_img_root = os.path.join(
        args.gt_output_dir, img_view_prefix, "undistorted", args.data_name
    )
    # Extract frames with annotations for all takes
    print("Undistorting Aria images...")

    if args.mp:
        available_cpus = multiprocessing.cpu_count()
        print(f"Using {available_cpus} number of CPUs")
        P = multiprocessing.Pool(available_cpus - 2)

        try:
            for _ in tqdm(P.imap_unordered(undistort_take_single, zip(args.take_folder_list,
                                                               [args] * len(args.take_folder_list),
                                                               [dist_img_root] * len(args.take_folder_list),
                                                               [undist_img_root] * len(args.take_folder_list))),
                          total=len(args.take_folder_list)):
                pass
        except Exception as error:
            print("Child process failed")
            import sys
            sys.exit(1)
    else:
        for take_idx, take_name in enumerate(args.take_folder_list):
            # print(f"[{take_idx+1}/{len(args.take_folder_list)}] processing {take_name}")
            undistort_take_single(take_name, args, dist_img_root, undist_img_root)

def decode_video(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # scale video down from 1440 x 1080 to 320 x 240
    # maintain highest quality with qscale:v 1
    # cmd = f"ffmpeg -loglevel error -i {video_path} -qscale:v 1 -vf scale=320:240 '{save_dir}/{video_name}_%10d.jpg' < /dev/null"

    cmd = f"ffmpeg -loglevel error -i {video_path} -qscale:v 1 -vf fps=10 '{save_dir}/%06d.jpg' < /dev/null"

    os.system(cmd)
    print(f"{video_path}, #frames={len(os.listdir(save_dir))}")

def get_take_from_take_json(takes_json_lst, take_name):
    try:
        return [_ for _ in takes_json_lst if _['take_name'] == take_name][0]
    except:
        import pdb
        pdb.set_trace()


def find_immediate_children_dirs(root_dir, leaf_directory_keyword=""):
    """
    leaf_directory_keyword: keyword that every leaf directory must have

    todo: this was changed to just the immediate children for speed
    """
    print("Finding children directories...")
    child_basenames = sorted(next(os.walk(root_dir))[1])
    return [os.path.join(root_dir, _) for _ in child_basenames]


def handle(input_tuple):
    resolution = 448
    take_video_dir, img_output_root, ego_aria_cam_name, take_name = input_tuple
    # take = get_take_from_take_json(takes_json_lst, take_name)

    # ego_aria_cam_name = get_ego_aria_cam_name(take)
    # Load current take's aria video

    if resolution == 448:
        curr_take_video_path = os.path.join(
            take_video_dir,
            take_name,
            "frame_aligned_videos/downscaled/448",
            f"{ego_aria_cam_name}_214-1.mp4",
        )
    elif resolution == 1404:
        curr_take_video_path = os.path.join(
            take_video_dir,
            take_name,
            "frame_aligned_videos",
            f"{ego_aria_cam_name}_214-1.mp4",
        )
    else:
        raise NotImplementedError

    curr_take_img_output_path = os.path.join(img_output_root, take_name)
    os.makedirs(curr_take_img_output_path, exist_ok=True)

    print(f"Decoding to: {curr_take_img_output_path}")
    decode_video(curr_take_video_path, curr_take_img_output_path)

def extract_aria_img(args):
    # Load all takes metadata
    takes_json_lst = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))
    img_view_prefix = "image_portrait_view" if args.portrait_view else "image"
    dist_img_root = os.path.join(
        args.gt_output_dir, img_view_prefix, "distorted", args.data_name
    )
    undist_img_root = os.path.join(
        args.gt_output_dir, img_view_prefix, "undistorted", args.data_name
    )
    # Extract frames with annotations for all takes
    print("Undistorting Aria images...")
    # for i, (take_uid, take_anno) in enumerate(gt_anno.items()):

    # Input and output root path
    take_video_dir = os.path.join(args.ego4d_data_dir, "takes")
    img_view_prefix = "image_portrait_view" if args.portrait_view else "image"
    img_output_root = os.path.join(
        args.gt_output_dir, img_view_prefix, "distorted", args.data_name
    )
    os.makedirs(img_output_root, exist_ok=True)

    if args.mp:
        # make this 128 after I get it working locally
        available_cpus = multiprocessing.cpu_count()

        print(f"Using {available_cpus} number of CPUs")
        P = multiprocessing.Pool(available_cpus - 2)
        # P.map(handle, mp4_clip_lst)

        # take = get_take_from_take_json(takes_json_lst, take_name)
        #
        # ego_aria_cam_name = get_ego_aria_cam_name(take)

        ego_aria_cams = [get_ego_aria_cam_name(get_take_from_take_json(takes_json_lst, take_name)) for take_name in args.take_folder_list]
        for _ in tqdm(P.imap_unordered(handle, zip([take_video_dir] * len(args.take_folder_list),
                                                   [img_output_root] * len(args.take_folder_list),
                                                   ego_aria_cams,
                                                   args.take_folder_list)),
                      total=len(args.take_folder_list)):
            pass
    else:
        # single threads
        for take_idx, take_name in enumerate(args.take_folder_list):

            # Extract frames with annotations for all takes
            print("Extracting Aria images...")
            # Get current take's metadata

            take = get_take_from_take_json(takes_json_lst, take_name)
            print(f"[{take_idx+1}/{len(args.take_folder_list)}] processing {take_name}")
            ego_aria_cam_name = get_ego_aria_cam_name(take)
            # Load current take's aria video

            if args.resolution == 448:
                curr_take_video_path = os.path.join(
                    take_video_dir,
                    take_name,
                    "frame_aligned_videos/downscaled/448",
                    f"{ego_aria_cam_name}_214-1.mp4",
                )
            elif args.resolution == 1404:
                curr_take_video_path = os.path.join(
                    take_video_dir,
                    take_name,
                    "frame_aligned_videos",
                    f"{ego_aria_cam_name}_214-1.mp4",
                )
            else:
                raise NotImplementedError

            if not os.path.exists(curr_take_video_path):
                print(
                    f"[Warning] No frame aligned videos found at {curr_take_video_path}. Skipped take {take_name}."
                )
                continue
            curr_take_img_output_path = os.path.join(img_output_root, take_name)
            os.makedirs(curr_take_img_output_path, exist_ok=True)

            print(f"Decoding to: {curr_take_img_output_path}")
            decode_video(curr_take_video_path, curr_take_img_output_path)


def save_test_gt_anno(output_dir, gt_anno_private):
    # 1. Save private annotated test JSON file
    with open(
        os.path.join(output_dir, f"ego_pose_gt_anno_test_private.json"), "w"
    ) as f:
        json.dump(gt_anno_private, f)
    # 2. Exclude GT 2D & 3D joints and valid flag information for public un-annotated test file
    gt_anno_public = copy.deepcopy(gt_anno_private)
    for _, take_anno in gt_anno_public.items():
        for _, frame_anno in take_anno.items():
            for k in [
                "left_hand_2d",
                "right_hand_2d",
                "left_hand_3d",
                "right_hand_3d",
                "left_hand_valid_3d",
                "right_hand_valid_3d",
            ]:
                frame_anno.pop(k)
    # 3. Save public un-annotated test JSON file
    with open(os.path.join(output_dir, f"ego_pose_gt_anno_test_public.json"), "w") as f:
        json.dump(gt_anno_public, f)


def create_gt_anno(args):
    """
    Creates ground truth annotation file for train, val and test split. For
    test split creates two versions:
    - public: doesn't have GT 3D joints and valid flag information, used for
    public to do local inference
    - private: has GT 3D joints and valid flag information, used for server
    to evaluate model performance
    """
    print("Generating ground truth annotation files...")
    for anno_type in args.anno_types:
        for split in args.splits:
            # For test split, only save manual annotation
            if split == "test" and anno_type == "auto":
                print("[Warning] No test gt-anno will be generated on auto data. Skipped for now.")
            # Get ground truth annotation
            gt_anno = ego_pose_anno_loader(args, split, anno_type)
            gt_anno_output_dir = os.path.join(
                args.gt_output_dir, "annotations", anno_type
            )
            os.makedirs(gt_anno_output_dir, exist_ok=True)
            # Save ground truth JSON file
            if split in ["train", "val"]:
                with open(
                    os.path.join(
                        gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json"
                    ),
                    "w",
                ) as f:
                    json.dump(gt_anno.db, f)
            # For test split, create two versions of GT-anno
            else:
                if len(gt_anno.db) == 0:
                    print(
                        "[Warning] No test gt-anno will be generated. Please download public release from shared drive."
                    )
                else:
                    save_test_gt_anno(gt_anno_output_dir, gt_anno.db)

import re
def extract_aria_calib_to_json_pyvrs(vrs_path, output_path):
    def extract_calib_json(text):
        # Find the start of the JSON content
        start_match = re.search(r'calib_json\s*\|\s*{', text)
        if not start_match:
            return None

        start_index = start_match.end() - 1  # Include the opening brace

        # Initialize brace counter and result
        brace_count = 0
        result = ""

        # Iterate through the text from the start index
        for i in range(start_index, len(text)):
            char = text[i]
            result += char

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

                # If we've found the matching closing brace, we're done
                if brace_count == 0:
                    return result

        # If we've reached here, we didn't find a matching closing brace
        return None

    loaded = pyvrs.reader.AsyncVRSReader(vrs_path)
    json_text = extract_calib_json(str(loaded))

    with open(output_path, "w") as f:
        f.write(json_text)


def create_single_json(input_tuple):
    take_name, take, aria_calib_json_output_dir = input_tuple
    # Get aria name
    # assert len(take) == 1, f"Take: {take_name} can't be found in takes.json"
    aria_cam_name = get_ego_aria_cam_name(take)
    # 1. Generate aria calib JSON file
    vrs_path = os.path.join(
        args.ego4d_data_dir,
        "takes",
        take_name,
        f"{aria_cam_name}_noimagestreams.vrs",
    )
    if not os.path.exists(vrs_path):
        print(
            f"[Warning] No take vrs found at {vrs_path}. Skipped take {take_name}."
        )
        return None

    output_path = os.path.join(aria_calib_json_output_dir, f"{take_name}.json")

    if os.path.exists(output_path):
        print(f"Exists, skipping: {output_path}")
        return None

    extract_aria_calib_to_json_pyvrs(vrs_path, output_path)
    # 2. Overwrite f, cx, cy parameter from JSON file

    aria_calib_json = json.load(open(output_path))

    # Overwrite f, cx, cy
    # then, resave it at the same path
    all_cam_calib = aria_calib_json["CameraCalibrations"]
    aria_cam_calib = [c for c in all_cam_calib if c["Label"] == "camera-rgb"][0]

    # TODO: this will be changed later if we resize the input
    aria_cam_calib["Projection"]["Params"][0] /= 2
    aria_cam_calib["Projection"]["Params"][1] = (
                                                        aria_cam_calib["Projection"]["Params"][1] - 0.5 - 32
                                                ) / 2
    aria_cam_calib["Projection"]["Params"][2] = (
                                                        aria_cam_calib["Projection"]["Params"][2] - 0.5 - 32
                                                ) / 2
    # Save updated JSON calib file

    print("Saving new aria json calib")
    print(output_path)
    with open(output_path, "w") as f:
        json.dump(aria_calib_json, f)

def create_aria_calib(args):
    # Get all annotated takes
    # all_local_take_uids = find_annotated_takes(
    #     args.ego4d_data_dir, args.splits, args.anno_types
    # )

    splits_json = json.load(open(os.path.join(args.ego4d_data_dir, "annotations/splits.json")))

    takes_json_lst = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

    def get_take_uid_from_take_name(take_name):
        return [take['take_uid'] for take in takes_json_lst if take['take_name'] == take_name][0]

    if args.split:
        split_uids = splits_json['split_to_take_uids'][args.split]
    else:
        split_uids = [get_take_uid_from_take_name(take_name) for take_name in args.take_folder_list]

    # Create aria calib JSON output directory
    aria_calib_json_output_dir = os.path.join(args.gt_output_dir, "aria_calib_json")
    os.makedirs(aria_calib_json_output_dir, exist_ok=True)
    # Find uid and take info
    takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))
    take_to_uid = {
        each_take["take_name"]: each_take["take_uid"]
        for each_take in takes
        if each_take["take_uid"] in split_uids
    }
    assert len(split_uids) == len(
        take_to_uid
    ), "Some annotation take doesn't have corresponding info in takes.json"
    # Export aria calibration to JSON file
    print("Generating aria calibration JSON file...")

    if args.mp:
        takes_by_take_name = [[t for t in takes if t["take_name"] == take_name][0] for take_name in args.take_folder_list]

        # TODO: if already exists, don't load vrs
        available_cpus = multiprocessing.cpu_count()
        print(f"Using {available_cpus} number of CPUs")
        P = multiprocessing.Pool(available_cpus - 2)

        for _ in tqdm(P.imap_unordered(create_single_json, zip(args.take_folder_list,
                                                          takes_by_take_name,
                                                           [aria_calib_json_output_dir] * len(args.take_folder_list))),
                      total=len(args.take_folder_list)):
            pass

    else:
        for take_name, take_uid in tqdm(take_to_uid.items()):
            take = [t for t in takes if t["take_name"] == take_name][0]
            create_single_json([take_name, take, aria_calib_json_output_dir])

def main(args):
    for step in args.steps:
        if step == "create_aria_calib":
            create_aria_calib(args)
        elif step == "gt_anno":
            create_gt_anno(args)
        elif step == "extract_aria_img":
            extract_aria_img(args)
        elif step == "undistort_aria_img":
            undistort_aria_img(args)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Ego-pose baseline model dataset preparation")

    parser.add_argument(
        "--split",
        type=str,
        # nargs="+",
        # default=["train", "val", "test"],
        help="train/val/test split of the dataset",
    )

    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["aria_calib", "gt_anno", "raw_image", "undistorted_image"],
        help="""
            Determine which step should be executed in data preparation:
            - aria_calib: Generate aria calibration JSON file for easier loading
            - gt_anno: Extract ground truth annotation file
            - raw_image: Extract raw ego-view (aria) images
            - undistorted_image: Undistort raw aria images
            """,
    )

    parser.add_argument(
        "--anno_types",
        type=str,
        nargs="+",
        default=["manual"],
        help="Type of annotation: use manual or automatic data",
    )

    # Ego4d data and output directory
    parser.add_argument(
        "--ego4d_data_dir",
        type=str,
        default=None,
        help="Directory of downloaded Ego4D data, including annotations, captures, takes, metadata.",
        required=True,
    )
    parser.add_argument(
        "--gt_output_dir",
        type=str,
        default=None,
        help="Directory to store preprocessed ground truth annotation JSON file",
        required=True,
    )
    parser.add_argument(
        "--portrait_view",
        action="store_true",
        help="whether hand keypoints and images stay in portrait view, instead of landscape view (default)",
    )

    # Threshold and parameters in dataloader
    parser.add_argument("data_name", help="Name of this dataset")

    parser.add_argument("--valid_kpts_num_thresh", type=int, default=10)
    parser.add_argument("--bbox_padding", type=int, default=20)
    parser.add_argument("--reproj_error_threshold", type=int, default=30)

    parser.add_argument("--meta_folder", help="Folder of takes")

    parser.add_argument("--take_folder_list",
                        type=str,
                        nargs="+",
                        help="List of specific takes we want to run things on. Not the full path, just the take names")

    parser.add_argument("--resolution",
                        type=int,
                        default=[448, 1404])

    parser.add_argument("--mp",
                        action="store_true",
                        help="Do multiprocessing")
    args = parser.parse_args()

    if args.split:
        print("Split specified, scanning for take list")
        print(args.split)

        # /data/pulkitag/models/rli14/data/egoexo/annotations/splits.json

        splits_json = json.load(open(os.path.join(args.ego4d_data_dir, "annotations/splits.json")))

        split_uids = splits_json['split_to_take_uids'][args.split]

        takes_json_lst = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

        def get_take_name_from_take_uid(take_uid):
            return [take['take_name'] for take in takes_json_lst if take['take_uid'] == take_uid][0]

        split_uids_to_take_names = [get_take_name_from_take_uid(tuid) for tuid in split_uids]
        args.take_folder_list = split_uids_to_take_names
    main(args)
