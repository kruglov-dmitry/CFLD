# from mmpose.apis import MMPoseInferencer
#
# img_path = './images_to_play_with/fd_sm.jpg'
#
# # instantiate the inferencer using the model alias
# inferencer = MMPoseInferencer('human')
#
# # The MMPoseInferencer API employs a lazy inference approach,
# # creating a prediction generator when given input
# result_generator = inferencer(img_path, show=True)
# result = next(result_generator)

from mmpose.apis import inference_topdown, init_model, inference_bottomup
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import cv2

import mmcv
from mmcv import imread
from PIL import Image
import numpy as np

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

# config_file = 'td-hm_hrnet-w48_8xb64-20e_posetrack18-256x192.py'
# checkpoint_file = 'hrnet_w48_posetrack18_256x192-b5d9b3f1_20211130.pth'

# config_file = 'rtmpose-l_8xb512-700e_body8-halpe26-256x192.py'
# checkpoint_file = 'rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth'

# config_file = 'rtmpose-l_8xb256-420e_body8-256x192.py'
# checkpoint_file = 'rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'

pose_estimator = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# img_path = './images_to_play_with/fd_sm_tiny.jpg'
# img_path = './images_to_play_with/target_pose_2_sm.jpeg'
# img_path = './images_to_play_with/target_pose_3.jpeg'
# img_path = './images_to_play_with/target_pose_4.jpeg'
# img_path = './images_to_play_with/target_pose_5.jpeg'
img_path = './images_to_play_with/target_pose_6.jpeg'
# img_path = './images_to_play_with/target_pose_1.jpeg'
out_file = 'output.jpg'

# please prepare an image with person
pose_results = inference_topdown(pose_estimator, img_path)
#pose_results = inference_bottomup(pose_estimator, img_path)
print(pose_results)

import torch
from pose_utils import (cords_to_map, draw_pose_from_cords,
                        load_pose_cords_from_strings)


def build_pose_img(some_array, new_size=(256, 256) , old_size=(256, 176)):
    # fd_sm_tiny.jpg
    # array = load_pose_cords_from_strings(
    #     "[46, 44, 39, 51, 41, 84, 86, 121, 131, 126, 121, 166, 164, 251, 249, 241, 241, -1]",
    #     "[74, 82, 69, 87, 59, 99, 39, 109, 32,  94,  44,  87,  49,  74,  52,  72,  59, -1]",
    #
    # )
    # array = load_pose_cords_from_strings(
    #     "[36, 32, 29, 36, 29, 49, 49, 89, 89, 116, 119, 86, 82, 129, 132, 132, 136, -1]",
    #     "[176, 179, 173, 173, 159, 149, 159, 156, 159, 159, 156, 103, 119, 113, 143, 89, 99, -1]",
    # )

    # array = load_pose_cords_from_strings(
    #     "[25, 50, 50, 86, 125, 49, 84, 115, 115, 167, 226, 116, 171, 208, 20, 20, 24, 23]",
    #  "[88, 91, 72, 64, 54, 110, 117, 119, 74, 68, 66, 100, 100, 118, 84, 94, 80, 101]"
    # )

    print("ORIG")
    print(some_array)
    rearraned = rearrange_keypoints(some_array)
    print("REARRANGED")
    print(rearraned)
    x_coords, y_coords = extract_coordinates(rearraned)
    print("COORDINATES")
    print("Y\n", [int(x) for x in y_coords])
    print("X\n", [int(x) for x in x_coords])

    array = load_pose_cords_from_strings(
        "[" + ",".join([str(int(x)) for x in y_coords]) + "]", # y
      "[" + ",".join([str(int(x)) for x in x_coords]) + "]" # x
    )

    pose_map = torch.tensor(cords_to_map(array, new_size, old_size).transpose(2, 0, 1), dtype=torch.float32)
    pose_img = torch.tensor(draw_pose_from_cords(array, new_size, old_size).transpose(2, 0, 1) / 255., dtype=torch.float32)
    pose_img = torch.cat([pose_img, pose_map], dim=0)
    return pose_img

def rearrange_keypoints(hrnet_keypoints):
    # nose,
    # neck, left shoulder, left elbow, left arm, right shoulder, right elbow, right arm
    # left_leg_start, left_knee, left_foot, right_leg_start, right_knee, right_foot,
    # left eye, right eye, left ear, right ear

    ochuman_keypoints = [
        hrnet_keypoints[0],  # nose
        (hrnet_keypoints[5] + hrnet_keypoints[6]) / 2,  # neck
        hrnet_keypoints[5],  # left shoulder
        hrnet_keypoints[7],  # left elbow
        hrnet_keypoints[9],  # left wrist
        hrnet_keypoints[6],  # right shoulder
        hrnet_keypoints[8],  # right elbow
        hrnet_keypoints[10], # right wrist
        hrnet_keypoints[11], # left hip
        hrnet_keypoints[13], # left knee
        hrnet_keypoints[15], # left ankle
        hrnet_keypoints[12], # right hip
        hrnet_keypoints[14], # right knee
        hrnet_keypoints[16], # right ankle
        hrnet_keypoints[1],  # left eye
        hrnet_keypoints[2],  # right eye
        hrnet_keypoints[3],  # left ear
        hrnet_keypoints[4],  # right ear
    ]
    return ochuman_keypoints

def extract_coordinates(ochuman_keypoints):
    x_coords = [point[0] for point in ochuman_keypoints]
    y_coords = [point[1] for point in ochuman_keypoints]
    return x_coords, y_coords



pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

data_samples = merge_data_samples(pose_results)

# show the results
img = mmcv.imread(img_path, channel_order='rgb')
show_interval=0
visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=True,
        draw_bbox=True,
        show=False,
        wait_time=show_interval,
        out_file=out_file,
        kpt_thr=0.3)
vis_result = visualizer.get_image()
vis_result_prep = vis_result[:,:,::-1].copy()

pose_img_tensor = build_pose_img(
    pose_results[0].pred_instances.get('keypoints')[0],
    #new_size=#vis_result_prep.shape[:2],
    old_size=img.shape[:2]
    #old_size=(256, 171)#vis_result_prep.shape[:2][::-1]#   (256, 256)
).unsqueeze(0)
new_img = Image.fromarray((pose_img_tensor[0][:3].permute((1, 2, 0)) * 255.).long().numpy().astype(np.uint8))
open_cv_image = np.array(new_img)
# Convert RGB to BGR
open_cv_image = open_cv_image[:, :, ::-1].copy()

# height_diff = vis_result_prep.shape[0] - open_cv_image.shape[0]
# top_padding = height_diff // 2
# bottom_padding = height_diff - top_padding
#
# # Calculate horizontal padding
# width_diff = vis_result_prep.shape[1] - open_cv_image.shape[1]
# left_padding = abs(width_diff) // 2
# right_padding = abs(width_diff) - left_padding
#
# print(open_cv_image.shape)
# # Add padding (black border in this example)
# open_cv_image = cv2.copyMakeBorder(open_cv_image, abs(top_padding), abs(bottom_padding), left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)

print(open_cv_image.shape)
cv2.imshow("U", open_cv_image)
cv2.waitKey(0)
#
# print(vis_result_prep.shape)
# cv2.imshow("U", vis_result_prep)
# cv2.waitKey(0)

horizontal_concat = np.concatenate((vis_result_prep, open_cv_image), axis=1)


cv2.imshow("U", horizontal_concat)
cv2.waitKey(0)

# print(pose_results[0].pred_instances.get('keypoints')[0])
# print(pose_results[0].pred_instances.keypoints_visible[0])
# print(len(pose_results[0].pred_instances.keypoints[0]))
# print(len(pose_results[0].pred_instances.keypoints_visible[0]))


cv2.destroyAllWindows()



# Target pose 3
# Y [51, 73, 71, 96, 116, 74, 104, 121, 129, 134, 204, 131, 136, 204, 49, 49, 51, 51]
# X [83, 85, 108, 125, 128, 63, 50, 43, 103, 133, 138, 73, 38, 38, 88, 78, 93, 73]

# Target pose 2
# Y [54, 74, 74, 135, 176, 74, 135, 181, 130, 196, 201, 125, 201, 206, 49, 44, 49, 44]
# X [267, 235, 227, 237, 242, 242, 242, 237, 156, 171, 136, 181, 217, 151, 273, 262, 257, 242]

# Target pose 1
# Y [94, 100, 109, 134, 154, 91, 81, 69, 146, 189, 229, 134, 101, 144, 91, 89, 91, 84]
# X [116, 99, 111, 119, 131, 86, 61, 41, 81, 81, 79, 71, 54, 44, 119, 116, 119, 109]


# Target pose 4
# Y [85, 75, 70, 130, 195, 80, 125, 170, 90, 135, 140, 115, 130, 145, 80, 90, 60, 95]
# X [94, 127, 139, 139, 124, 114, 104, 99, 225, 285, 340, 210, 280, 330, 89, 84, 99, 84]

# Target pose 5
# Y [64, 79, 74, 54, 54, 84, 94, 79, 136, 179, 194, 136, 171, 206, 61, 61, 61, 61]
# X [81, 83, 96, 116, 101, 69, 56, 49, 91, 91, 54, 76, 104, 96, 84, 76, 89, 71]

# target pose 6
# Y [79, 107, 107, 139, 148, 107, 134, 144, 157, 162, 167, 157, 162, 162, 74, 74, 79, 79]
# X [174, 174, 197, 202, 197, 151, 147, 147, 188, 234, 285, 160, 114, 59, 179, 165, 183, 160]

