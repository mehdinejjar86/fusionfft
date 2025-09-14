import tifffile
import cv2
import os
from torch.nn import functional as F

def read_image(image_path, extension):
  flag_cv2 = False

  if extension == '.tif':
    image = tifffile.imread(image_path)
  else:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    flag_cv2 = True
  if len(image.shape) < 3:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  if flag_cv2:
    image = image[:, :, ::-1].copy()  # Convert BGR to RGB
  return image

def save_image(tensor, output_path, name, extension, h, w, dtype, max_val):
    
  image = (tensor[0] * max_val).cpu().detach().numpy().astype(dtype).transpose(1, 2, 0)[:h, :w]

  output_path = os.path.join(output_path, f"{name:0>7d}{extension}")

  if extension == '.tif':
      tifffile.imwrite(output_path, image[:, :, ::-1])
  else:
      cv2.imwrite(output_path, image[:, :, ::-1])  # Convert RGB back to BGR for saving

def pad_image(img, padding):
        return F.pad(img, padding)