import asyncio
import os
import time
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from diffusers import DiffusionPipeline
import torch
import ImageReward as reward
import random
import redis

redis_async_result = RedisAsyncResultBackend(
    redis_url="redis://localhost:6379",
)

# Or you can use PubSubBroker if you need broadcasting
broker = ListQueueBroker(
    url="redis://localhost:6379",
    result_backend=redis_async_result,
)
prompts = [
    "A futuristic cityscape with flying cars and neon signs in the rain.",
    "A serene forest with ancient trees and a carpet of bluebells.",
    "An astronaut playing chess with an alien on Mars.",
    "A steampunk airship docked at a floating island.",
    "A medieval castle surrounded by a moat with dragons flying overhead.",
    "A post-apocalyptic wasteland with nature reclaiming a crumbling city.",
    "A magical library with floating books and glowing orbs of light.",
    "A cyberpunk street market bustling with robots and holograms.",
    "An underwater city with merpeople and bioluminescent coral.",
    "A giant robot battling a monstrous kaiju in the middle of Tokyo.",
    "A hidden garden with a crystal-clear waterfall and mystical creatures.",
    "A space station orbiting a vibrant alien planet.",
    "A Victorian-era detective's office with mysterious artifacts.",
    "A desert oasis at sunset with camels and nomadic tents.",
    "A pirate ship sailing through the sky among the clouds.",
    "A snowy village with cozy cabins and a northern lights display.",
    "A superhero showdown in a modern metropolis.",
    "A tranquil Zen garden with cherry blossoms and a koi pond.",
    "A jazz club in the 1920s with flappers and musicians.",
    "A high-speed train zooming through a futuristic landscape.",
    "A haunted mansion with ghosts and eerie candlelight.",
    "A Viking longship navigating through icy fjords.",
    "A bustling Renaissance fair with jesters, knights, and artisans.",
    "A lush rainforest with exotic birds and ancient ruins.",
    "A neon-lit arcade in the 1980s with classic video games."
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
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
print("sdxl model is loaded")
scoring_model = reward.load("ImageReward-v1.0")
print("Scoring model loaded.")
@broker.task
async def generate_image(prompt: str, guidance_scale: int):
    global result
    """Solve all problems in the world."""

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
    # await asyncio.sleep(5.5)

    print(f"-------------prompt in broker: {prompt}-------------------")
    print(f"-------------Guidance_scale in broker: {guidance_scale}-------------------")
    
    # time.sleep(30)
    pipe.to("cuda")    
    # _pipe = pipe.to(f"cuda:{gpu_index}")
    images = pipe(prompt=prompt).images
    print("Successfully generated images.")
    score = scoring_model.score(prompt, images)
    # try:
    #     os.mkdir(f"{hash_function(prompt)}")
    # except:
    #     pass
    # images[0].save(f"{hash_function(prompt)}/{score}.png")

    # result.append({"image": images, "score": score})
    # print(result)
    print("All problems are solved!")
    # return images, score
    return {"prompt": prompt, "score": score}

async def main():
   
    await broker.startup()
    r = redis.Redis(host='localhost', port=6379, db=0)

    while True:
        global result
        random_int = random.randint(0, 20)
        prompt = prompts[random_int]
        start_time = time.time()
        print(f"prompt: {prompt}")
        num_images = 3
        tasks = []
        results = []
        for i in range(num_images):
            guidance_scale = random.uniform(5, 10)
            task = await generate_image.kiq(prompt, guidance_scale)
            tasks.append(task)
            # print(task)
            # task.wait_result()
            # result.append(await task.get_result())
            # print(await task.wait_result())
            # await task
        for task in tasks:
            result = await task.wait_result()            
            results.append(result)
        print("----------------Result------------------------")
        print(results)
        # Note: check if all of {num_images} images are generated
        
        end_time = time.time()
        print(f"All of {num_images} images are generated in {end_time-start_time} seconds.")
        # time.sleep(30)
        
        r.flushdb()
        print(r.dbsize())

if __name__ == "__main__":
    asyncio.run(main())
