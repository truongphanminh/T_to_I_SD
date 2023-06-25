#mã tổng
%pip install --quiet --upgrade diffusers transformers accelerate mediapy triton scipy ftfy spacy
!pip install xformers # (The xformers package is mandatory to be able to create several 768x768 images)

model_id = "stabilityai/stable-diffusion-2-1-base"
model_id = "stabilityai/stable-diffusion-2-1"
model_id = "wavymulder/portraitplus"

from diffusers import PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler

scheduler = None
# scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
# scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
# scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

import mediapy as media

from diffusers import StableDiffusionPipeline
import random
from IPython.display import clear_output

from google.colab import drive
import torch
from PIL import Image
#from diffusers import StableDiffusionImg2ImgPipeline
from IPython.display import display
from diffusers import StableDiffusionImg2ImgPipeline as SDIIP
from IPython.display import display, Image as IPImage

device = "cuda"

model_id_or_path = "dreamlike-art/dreamlike-photoreal-2.0" #@param ["runwayml/stable-diffusion-v1-5", "dreamlike-art/dreamlike-photoreal-2.0", "stabilityai/stable-diffusion-2-1-base", "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"]

pipe_kwargs = {"torch_dtype": torch.float16}
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()


prompt = input("Nhập prompt: ")

remove_safety = False

if remove_safety:
  negative_prompt = None
  pipe.safety_checker = None
else:
  negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,<bad-image-v2-39000:0.8>,<bad_prompt_version2:0.8>,<negative_hand-neg:0.8>,<easynegative:0.8>,<Unspeakable-Horrors-Composition-4:0.8>,(selfie:1.2),(visible nipples:1.1),sunglasses,(pubic hair),3d rendering,cgi,sketch,painting,(extra fingers,deformed hands,polydactyl),lowres, bad perspective, bad proportions, unrealistic, text, error, missing details, extra details, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, not symmetrical, not balanced, distorted, deformed"

import random
from google.colab import files
import ipywidgets as widgets
from IPython.display import display

remove_safety = False
num_images = 4
seed = random.randint(0, 2147483647)
images = pipe(
    prompt,
    height=768,
    width=512,
    num_inference_steps = 25,
    guidance_scale = 9,
    num_images_per_prompt = num_images,
    negative_prompt = negative_prompt,
    generator = torch.Generator("cuda").manual_seed(seed)
    ).images



media.show_images(images)
display(f"Seed: {seed}")
images[0].save("output.jpg")
