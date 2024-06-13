from diffusers import DiffusionPipeline
import torch
import ImageReward as reward
import time

def generate_image(prompt: str):
    print(f"------------------{torch.cuda.current_device()}--------------------")

    scoring_model = reward.load("ImageReward-v1.0")
    time.sleep(5)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    images = pipe(prompt=prompt).images[0]
    score = scoring_model.score(prompt, images)
    return images, score
    # image.save("sample.png")
if __name__ == '__main__':
    while True:
        images, score = generate_image("A cyberpunk street market bustling with robots and holograms.")
        print(score)
