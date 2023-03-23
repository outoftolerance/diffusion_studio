from PySide6.QtCore import QRunnable, Slot, QThreadPool

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionUpscalePipeline, DPMSolverMultistepScheduler

class UpscaleWorker(QRunnable):
    def __init__(self, image, scheduler, prompt, negative_prompt, guidance_scale, inference_step_count):
        super(UpscaleWorker, self).__init__()
        self._image = image.resize((512, 512))
        self._model = "stabilityai/stable-diffusion-x4-upscaler"
        self._scheduler = scheduler
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count

    @Slot()
    def run(self):
        #Setup pipeline
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(self._model, revision="fp16", torch_dtype=torch.float16)
        
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

        #Start generation
        print("Generating...")
        image = pipeline(
            image = self._image,
            prompt = self._prompt,
            negative_prompt = self._negative_prompt,
            guidance_scale = self._guidance_scale,
            num_inference_steps = self._inference_step_count,
            ).images[0]
        print("Done generating.")

        #Save output
        print("Saving image...")
        image_metadata = PngInfo()
        image_metadata.add_text("Scheduler", str(self._scheduler))
        image_metadata.add_text("Prompt", str(self._prompt))
        image_metadata.add_text("Negative Prompt", str(self._negative_prompt))
        image_metadata.add_text("Guidance Scale", str(self._guidance_scale))
        image_metadata.add_text("Inference Step Count", str(self._inference_step_count))
        image.save(f"image_upscale.png", pnginfo=image_metadata)
        print("Done saving.")
