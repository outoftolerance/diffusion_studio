from PySide6.QtCore import QRunnable, Slot, QThreadPool

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler

class DiffusionWorker(QRunnable):
    def __init__(self, model, scheduler, prompt, negative_prompt, seed, seed_lock, guidance_scale, inference_step_count, image_count=1):
        super(DiffusionWorker, self).__init__()
        self._model = model
        self._scheduler = scheduler
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._seed = seed
        self._seed_lock = seed_lock
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count
        self._image_count = image_count

    @Slot()
    def run(self):
        #Print config
        print(f"Model: { self._model }")
        print(f"Guidance Scale: { self._guidance_scale }")
        
        #Setup Pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(self._model, torch_dtype=torch.float16)

        #Determine Scheduler
        if self._scheduler == "EulerAncestralDiscreteScheduler":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        elif self._scheduler == "EulerDiscreteScheduler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        elif self._scheduler == "DDIMScheduler":
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        elif self._scheduler == "DDPMScheduler":
            pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        elif self._scheduler == "DPMSolverMultistepScheduler":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        elif self._scheduler == "DPMSolverSinglestepScheduler":
            pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)
        else:
            print(f"Pipeline not found! Defaulting to Euler Ancestral")
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

        #Send to GPU
        pipeline = pipeline.to("cuda")

        #Setup the seed
        generators = []

        #Choose to use the input seed or a random one
        if len(self._seed) > 0:
            seed = int(self._seed)
        else:
            seed = torch.Generator(device="cuda").seed()

        #Create the generators
        for i in range(0, self._image_count):
            #Check if seed should be locked for all generators
            if self._seed_lock:
                generators.append(torch.Generator(device="cuda").manual_seed(seed))
            else:
                generators.append(torch.Generator(device="cuda").manual_seed(seed + i))

        #Start generation
        print("Generating...")
        images = pipeline(
            prompt = self._prompt,
            negative_prompt = self._negative_prompt,
            generator = generators,
            guidance_scale = self._guidance_scale,
            num_inference_steps = self._inference_step_count,
            num_images_per_prompt = self._image_count,
            ).images
        print("Done generating.")

        #Save output
        for i in range(len(images)):
            print("Saving image...")
            image = images[i]
            image_metadata = PngInfo()
            image_metadata.add_text("Model", str(self._model))
            image_metadata.add_text("Scheduler", str(self._scheduler))
            image_metadata.add_text("Prompt", str(self._prompt))
            image_metadata.add_text("Negative Prompt", str(self._negative_prompt))
            image_metadata.add_text("Seed", str(generators[i].initial_seed()))
            image_metadata.add_text("Guidance Scale", str(self._guidance_scale))
            image_metadata.add_text("Inference Step Count", str(self._inference_step_count))
            image.save(f"output/image_{i}.png", pnginfo=image_metadata)
            print("Done saving.")
