from diffusers import AuraFlowPipeline
import torch

pipeline = AuraFlowPipeline.from_pretrained(
    "fal/AuraFlow-v0.3",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")


image = pipeline(
    prompt="rempage of the iguana character riding F1, fast and furious, cinematic movie poster",
    width=1024,
    height=1024,
    num_inference_steps=50, 
    generator=torch.Generator().manual_seed(1),
    guidance_scale=3.5,
).images[0]

image.save("output.png")