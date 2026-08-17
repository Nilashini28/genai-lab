from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cuda")

prompt = "A futuristic city at sunset"

image = pipe(prompt).images[0]

image.save("generated_image.png")

image.show()