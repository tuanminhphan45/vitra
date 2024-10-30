# Add Read HuggingFace Token to download weight models
from kaggle_secrets import UserSecretsClient
from huggingface_hub.hf_api import HfFolder

# Get HF Token, please add HF read token to Kaggle secret, or token = "hf..."
token = UserSecretsClient().get_secret("HF_TOKEN") 

HfFolder.save_token(token)

# Clone the repository and install the packages of vistral-V
!git clone https://github.com/hllj/Vistral-V.git
%cd Vistral-V
!pip install -e .

# Install customized transformers
!pip install transformers==4.41.2 -q

# Install stuff
!pip install ipykernel -q
!pip install ipywidgets -q
!jupyter nbextension enable --py widgetsnbextension

# Option 1: Lora Adapter and Base model
# model_path = "Vi-VLM/llava-vistral-7b-lora"
# model_base = "Viet-Mistral/Vistral-7B-Chat"

# Option 2: Full Model

model_path = "Vi-VLM/Vistral-V-7B"
model_base = None

conv_mode = "vistral"

image_file = "assets/images/example.jpeg"

temperature = 0.2
max_new_tokens = 512
load_8bit = False
load_4bit = False

debug = False
device = "cuda:0" # device = "cuda" if you want to inference with multiple GPU devices

from llava.utils import disable_torch_init

disable_torch_init()

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# Get the model name
model_name = get_model_name_from_path(model_path)

# FIXME: Vistral-V is need to be fixed, add prefix 'llava-'
if 'vistral-v' in model_name.lower():
    model_name = 'llava-' + model_name

# Load model, tokenizer and processor stuff
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=model_base,
    model_name=model_name,
    load_8bit=load_8bit,
    load_4bit=load_4bit,
    device=device
)

import requests
from PIL import Image
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

image = load_image(image_file)

print("This is the image using")
display(image)

from llava.conversation import conv_templates

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "vistral" in model_name.lower():
    conv_mode = "vistral"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

conv = conv_templates[conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ('user', 'assistant')
else:
    roles = conv.roles

import torch

from llava.conversation import SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from transformers import TextStreamer
from IPython.display import display

# Get the image size and díplay it
image_size = image.size

# Similar operation in model_worker.py
image_tensor = process_images([image], image_processor, model.config)
if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

while True:
    try:
        inp = input(f"{roles[0]}: ")
    except EOFError:
        inp = ""
    if not inp:
        print("exit...")
        break

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        image = None

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    if debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")



