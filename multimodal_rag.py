!pip install -q -U transformers==4.37.2
!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
!pip install -q git+https://github.com/openai/whisper.git
!pip install -q gradio
!pip install -q gTTS

import torch
from transformers import BitsAndBytesConfig, pipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})

import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS

from PIL import Image

image_path = "img.jpg"
image = Image.open((image_path))
image

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

import locale
print(locale.getlocale())  
# Run the pipeline
print(locale.getlocale())  

max_new_tokens = 200

prompt_instructions = """
Describe the image using as much detail as possible,
is it a painting, a photograph, what colors are predominant,
what is the image about?
"""

prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
for sent in sent_tokenize(outputs[0]["generated_text"]):
    print(sent)

warnings.filterwarnings("ignore")
