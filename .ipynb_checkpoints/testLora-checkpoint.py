import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_images(prompt, modifiers, negative_prompt, num_images, batch_size, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_id = "stabilityai/stable-diffusion-2-1"
    
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.load_lora_weights("LORA_RESULTS2", weight_name="pytorch_lora_weights.safetensors")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    name_offset = 0
    # Generate the specified number of images
    index = 0
    while index < num_images:
        current_modifier = modifiers[(index // 4) % len(modifiers)]
        full_prompt = f"{prompt}, {current_modifier}"
        
        images = pipe(full_prompt, negative_prompt=negative_prompt, num_images_per_prompt=batch_size, cross_attention_kwargs={"scale": -0.4}).images
        for j, image in enumerate(images):
            if index >= num_images:
                break
            image.save(os.path.join(output_folder, f"image_{index+1+name_offset}.png"))
            index += 1

# Example usage


def fulltaskworker(promptNumber):
    prompts = [
        "a photograph of an airplane", 
        "a photograph of an automobile", 
        "a photograph of a bird", 
        "a photograph of a cat",
        "a photograph of a deer", 
        "a photograph of a dog", 
        "a photograph of a frog", 
        "a photograph of a horse", 
        "a photograph of a ship", 
        "a photograph of a truck"
    ]
    modifiers = [
        ["aircraft", "airplane", "fighter", "flying", "jet", "plane"],
        ["family", "new", "sports", "vintage"], 
        ["flying", "in a tree", "indoors", "on water", "outdoors", "walking"], 
        ["indoors", "outdoors", "walking", "running", "eating", "jumping", "sleeping", "sitting"], 
        ["herd", "in a field", "in the forest", "outdoors", "running", "wildlife photography"], 
        ["indoors", "outdoors", "walking", "running", "eating", "jumping", "sleeping", "sitting"], 
        ["European", "in the forest", "on a tree", "on the ground", "swimming", "tropical", "wildlife photography"],
        ["herd", "in a field", "in the forest", "outdoors", "running", "wildlife photograpahy"], 
        ["at sea", "boat", "cargo", "cruise", "on the water", "river", "sailboat", "tug"], 
        ["18-wheeler", "car transport", "fire", "garbage", "heavy goods", "lorry", "mining", "tanker", "tow"]
    ]
    output_folder = [
        "loraData1/SD21Airplane",
        "loraData1/SD21Automobile",
        "loraData1/SD21Bird",
        "loraData1/SD21Cat",
        "loraData1/SD21Deer",
        "loraData1/SD21Dog",
        "loraData1/SD21Frog",
        "loraData1/SD21Horse",
        "loraData1/SD21Ship",
        "loraData1/SD21Truck" 
    ]
    negative_prompt = "blurry, distorted, low qualiaty, low resolution, pixelated, noisy, abstract, surreal, unrealistic, cartoonish, animated, illustrated, painted, drawn, sketch, sketchy, sketch-like, sketchbook, doodle"

    batch_size = 16
    num_images = 6000
    generate_images(prompts[promptNumber], modifiers[promptNumber], negative_prompt, num_images, batch_size, output_folder[promptNumber])

for j in range(10):
    fulltaskworker(j)

