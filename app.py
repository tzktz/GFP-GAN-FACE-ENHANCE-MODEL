import os
os.system("pip install gfpgan")
import gradio as gr
from PIL import Image
import torch
import cv2
import glob
import numpy as np
from basicsr.utils import imwrite
from gfpgan import GFPGANer

bg_upsampler = None

# set up GFPGAN restorer
restorer = GFPGANer(
    model_path='GFPGANCleanv1-NoCE-C2.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler)


def inference(img):
    input_img = cv2.imread(img, cv2.IMREAD_COLOR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=False, only_center_face=False, paste_back=True)

    return Image.fromarray(restored_faces[0][:, :, ::-1])


title = "GFP-GAN FACE ENHANCE MODEL"
gr.Interface(
    inference,
    [gr.inputs.Image(type="filepath", label="Input")],
    gr.outputs.Image(type="pil", label="Output"),
    title=title
).launch(enable_queue=True)
