from diffusers import DDPMScheduler
from defaults import pose_transfer_C as cfg
from pose_transfer_train import build_model
from diffusers.configuration_utils import register_to_config 
from models import UNet, VariationalAutoencoder
import torch
import os
import numpy as np
import pandas as pd
from pose_utils import (cords_to_map, draw_pose_from_cords,
                        load_pose_cords_from_strings)
import random
from PIL import Image
from torchvision import transforms
import copy
import cv2

#def build_pose_img(annotation_file, img_path):

"""
[176.33333   36.166668]
 [179.66667   32.833336]
 [173.        29.500002]
 [173.        36.166668]
 [159.66667   29.500002]
 [149.66667   49.5     ]
 [159.66667   49.5     ]
 [156.33333   89.5     ]
 [159.66667   89.5     ]
 [159.66667  116.166664]
 [156.33333  119.5     ]
 [103.        86.166664]
 [119.666664  82.833336]
 [113.       129.5     ]
 [143.       132.83333 ]
 [ 89.666664 132.83333 ]
 [ 99.666664 136.16667 ]
"""
def build_pose_img():
    array = load_pose_cords_from_strings(
        "[176, 179, 173, 173, 159, 149, 159, 156, 159, 159, 156, 103, 119, 113, 143, 89, 99]",
        "[36,  32,  29,  36,  29,  49,  49,  89,  89,  116, 119, 86,  82,  129, 132, 132, 136]"
    )
    pose_map = torch.tensor(cords_to_map(array, (256, 256), (256, 176)).transpose(2, 0, 1), dtype=torch.float32)
    pose_img = torch.tensor(draw_pose_from_cords(array, (256, 256), (256, 176)).transpose(2, 0, 1) / 255., dtype=torch.float32)
    pose_img = torch.cat([pose_img, pose_map], dim=0)
    return pose_img


noise_scheduler = DDPMScheduler.from_pretrained("pretrained_models/scheduler/scheduler_config.json")
vae = VariationalAutoencoder(pretrained_path="pretrained_models/vae").eval().requires_grad_(False)
model = build_model(cfg).eval().requires_grad_(False)
unet = UNet(cfg).eval().requires_grad_(False)

print(model.load_state_dict(torch.load(
    os.path.join("checkpoints", "pytorch_model.bin"), map_location="cpu"
), strict=False))
print(unet.load_state_dict(torch.load(
    os.path.join("checkpoints", "pytorch_model_1.bin"), map_location="cpu"
), strict=False))


#
#   This is where as I understand we load template for another position?
#
# test_pairs = os.path.join("fashion", "fasion-resize-pairs-test.csv")
# test_pairs = pd.read_csv(test_pairs)
# annotation_file = pd.read_csv(os.path.join("fashion", "fasion-resize-annotation-test.csv"), sep=':')
# annotation_file = annotation_file.set_index('name')
# random_index = random.choice(range(len(test_pairs)))
# img_from_path, img_to_path = test_pairs.iloc[random_index]["from"], test_pairs.iloc[random_index]["to"]
# img_from = Image.open(os.path.join("fashion", "test_highres", img_from_path)).convert("RGB")
# img_to = Image.open(os.path.join("fashion", "test_highres", img_to_path)).convert("RGB")

img_from_path = './images_to_play_with/fd_sm.jpg'
img_to_path = './images_to_play_with/target_pose_2_sm.jpeg'
img_from = Image.open(img_from_path).convert("RGB")
img_from.resize((256, 256))

trans = transforms.Compose([
    transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_from_tensor = trans(img_from).unsqueeze(0)
print(img_from_tensor.shape)



pose_img_tensor = build_pose_img().unsqueeze(0)
Image.fromarray((pose_img_tensor[0][:3].permute((1, 2, 0)) * 255.).long().numpy().astype(np.uint8))

with torch.no_grad():
    c_new, down_block_additional_residuals, up_block_additional_residuals = model({
        "img_cond": img_from_tensor, "pose_img": pose_img_tensor})
    noisy_latents = torch.randn((1, 4, 64, 64))
    weight_dtype = torch.float32
    bsz = 1

    c_new = torch.cat([c_new[:bsz], c_new[:bsz], c_new[bsz:]])
    down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample, sample]).to(dtype=weight_dtype) \
                                        for sample in down_block_additional_residuals]
    up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), torch.zeros_like(v), v]).to(dtype=weight_dtype) \
                                        for k, v in up_block_additional_residuals.items()}

    noise_scheduler.set_timesteps(cfg.TEST.NUM_INFERENCE_STEPS)
    for t in noise_scheduler.timesteps:
        inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents], dim=0)
        inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
        noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
            down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
            up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

        noise_pred_uc, noise_pred_down, noise_pred_full = noise_pred.chunk(3)
        noise_pred = noise_pred_uc + \
                        cfg.TEST.DOWN_BLOCK_GUIDANCE_SCALE * (noise_pred_down - noise_pred_uc) + \
                        cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_down)
        noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    sampling_imgs = vae.decode(noisy_latents) * 0.5 + 0.5 # denormalize
    sampling_imgs = sampling_imgs.clamp(0, 1)

pil_image = Image.fromarray((sampling_imgs[0] * 255.).permute((1, 2, 0)).long().cpu().numpy().astype(np.uint8)).resize((256, 256))

opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse clicked at:", x, y)
        cv2.destroyAllWindows()

# Display the image in a window using OpenCV
cv2.imshow("Image", opencv_image)

# Set the mouse callback function to the window
cv2.setMouseCallback("Image", mouse_callback)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
