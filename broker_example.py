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

@broker.task
async def generate_image(prompt: str):
    global result
    """Solve all problems in the world."""
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    print("sdxl model is loaded")
    scoring_model = reward.load("ImageReward-v1.0") 
    print("Scoring model loaded.")

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    gpu_index = int(cuda_visible_devices.split(",")[0])
    device = torch.device(f"cuda:{gpu_index}")

    print(f"------------------GPU index is {gpu_index} and {torch.cuda.current_device()}--------------------")
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
    # await asyncio.sleep(5.5)
    print(f"-------------prompt in broker: {prompt}-------------------")
    
    # time.sleep(30)
    print(f"cuda:{gpu_index}")
    pipe.to("cuda")    
    # _pipe = pipe.to(f"cuda:{gpu_index}")
    images = pipe(prompt=prompt).images[0]
    print("Successfully generated images.")
    score = scoring_model.score(prompt, images)
    
    result.append({"image": images, "score": score})
    print("All problems are solved!")
    # return images, score
    return score

    

async def main():
   
    await broker.startup()
    r = redis.Redis(host='localhost', port=6379, db=0)

    while True:
        global result
        
        torch.cuda.set_device(1)
        print(f"------------------{torch.cuda.current_device()} in main--------------------")
        random_int = random.randint(0, 20)
        prompt = prompts[random_int]
        print(f"prompt: {prompt}")
        for i in range(2):
            task = generate_image.kiq(prompt)
            await task
        print(f"result : {result}")
        time.sleep(80)
        print(f"result : {result}")
        result = []
        r.flushdb()
        print(r.dbsize())

if __name__ == "__main__":
    asyncio.run(main())
