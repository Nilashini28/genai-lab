from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
from PIL import Image
import requests

image_url = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"

raw_image = Image.open(
    requests.get(image_url, stream=True).raw
).convert("RGB")


# ---------- Image Captioning ----------

cap_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

inputs = cap_processor(
    images=raw_image,
    return_tensors="pt"
)

out = cap_model.generate(**inputs)

caption = cap_processor.decode(
    out[0],
    skip_special_tokens=True
)

print("Caption:", caption)


# ---------- Visual Question Answering ----------

vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

question = "What is in the image?"

vqa_inputs = vqa_processor(
    images=raw_image,
    text=question,
    return_tensors="pt"
)

vqa_output = vqa_model.generate(
    **vqa_inputs
)

answer = vqa_processor.decode(
    vqa_output[0],
    skip_special_tokens=True
)

print("Question:", question)
print("Answer:", answer)