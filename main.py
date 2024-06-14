import asyncio
import io
import os
import time
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
import base64
from diffusers import (
    DiffusionPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import torch
import ImageReward as reward
import random
import redis
import argparse
from transformers import CLIPImageProcessor

redis_async_result = RedisAsyncResultBackend(
    redis_url="redis://localhost:6379",
)

# Or you can use PubSubBroker if you need broadcasting
broker = ListQueueBroker(
    url="redis://localhost:6379",
    result_backend=redis_async_result,
)
prompts = [
    "A serene forest with ancient trees and a carpet of bluebells.",    
]
result = []
def hash_function(input_string: str):
    """
    A simple hash function that converts a string input into an integer hash value.
    
    Args:
        input_string (str): The input string to be hashed.
    
    Returns:
        int: The hash value of the input string.
    """
    hash_value = 0
    for char in input_string:
        hash_value = (hash_value * 31 + ord(char)) % 2**32
    return hash_value


t2i_model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
print("sdxl model is loaded")
scoring_model = reward.load("ImageReward-v1.0")
print("Scoring model loaded.")
t2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
    t2i_model.scheduler.config
)
t2i_model.load_lora_weights("checkpoint-4700", weight_name="pytorch_lora_weights.safetensors", adapter_name="imagerewward-lora")
t2i_model.set_adapters(["imagereward-lora"], adapter_weights=[0.7])

def base64_to_pil_image(base64_image):
    image = base64.b64decode(base64_image)
    image = io.BytesIO(image)
    image = Image.open(image)
    image = image.convert("RGB")
    return image


def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    if format not in ["JPEG", "PNG"]:
        format = "JPEG"
    image_stream = io.BytesIO()
    image = image.convert("RGB")
    image.save(image_stream, format=format)
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
    return base64_image

@broker.task
async def generate_image(prompt: str, guidance_scale: int, num_inference_steps: int):
    global result
    """Solve all problems in the world."""

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
    
    print(f"-------------prompt in broker: {prompt}-------------------")
    print(f"-------------Guidance_scale in broker: {guidance_scale}-------------------")
    
    t2i_model.to("cuda")

    # _pipe = pipe.to(f"cuda:{gpu_index}")
    start_time = time.time()
    images = t2i_model(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images
    end_time = time.time()

    print(f"Successfully generated images in {end_time-start_time} seconds.")
    score = scoring_model.score(prompt, images)
    print(type(images[0]))
    
    # Note: encode <class 'PIL.Image.Image'>
    base64_image = pil_image_to_base64(images[0])
    # base64_image = base64.b64encode(image_bytes)
    print(type(base64_image))
    print("All problems are solved!")
    # return images, score
    return {"prompt": prompt, "score": score, "image": base64_image}

async def main():
   
    await broker.startup()
    r = redis.Redis(host='localhost', port=6379, db=0)

    while True:
        global result
        random_int = random.randint(0, 20)
        # prompt = prompts[random_int]
        prompt = prompts[0]
        start_time = time.time()
        print(f"prompt: {prompt}")
        num_images = 8
        tasks = []
        results = []
        for i in range(num_images):
            guidance_scale = random.uniform(5, 10)
            task = await generate_image.kiq(prompt, guidance_scale, 35)
            tasks.append(task)
        
        generated_images_number = 0
        for task in tasks:
            
            tmp_time = time.time()
            if tmp_time - start_time > 60:
                break
            result = await task.wait_result()
            
            results.append(result.return_value)
            generated_images_number += 1

        print("----------------Result------------------------")
        highest_score = max(results, key=lambda x: x["score"])["score"]
        print(highest_score)
        highest_image = max(results, key=lambda x: x["score"])["image"]
        image = base64_to_pil_image(highest_image)
        print(type(image))
        image.save(f"{highest_score}.png")
        # print(highest_image)
        # print(results[generated_images_number-1])
        # Note: check if all of {num_images} images are generated
        
        end_time = time.time()
        print(f"All of {generated_images_number} images are generated in {end_time-start_time} seconds.")
        # time.sleep(30)
        
        r.flushdb()
        print(f"size of redis db is {r.dbsize()}")

if __name__ == "__main__":
    asyncio.run(main())
