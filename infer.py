import torch
from prior_depth_anything import PriorDepthAnything

device = "cuda:0" if torch.cuda.is_available() else "cpu"
priorda = PriorDepthAnything(device=device, frozen_model_size='vitl', conditioned_model_size='vitb')
image_path = 'assets/sample-2/rgb.jpg'
prior_path = 'assets/sample-2/prior_depth.png'

output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=False)

print(output.shape)