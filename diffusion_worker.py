from PySide6.QtCore import QRunnable, Slot, QThreadPool

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class DiffusionWorker(QRunnable):
    def __init__(self, model, prompt, negative_prompt, guidance_scale, inference_step_count, image_count=1):
        super(DiffusionWorker, self).__init__()
        self._model = model
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count
        self._image_count = image_count

    @Slot()
    def run(self):
        #Print config
        print(f"Model: { self._model }")
        print(f"Guidance Scale: { self._guidance_scale }")
        
        #Setup pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(self._model, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")

        #Start generation
        print("Generating...")
        images = pipeline(
            prompt = [self._prompt] * self._image_count,
            negative_prompt = [self._negative_prompt] * self._image_count,
            guidance_scale = self._guidance_scale,
            num_inference_steps = self._inference_step_count,
            ).images
        print("Done generating.")

        #Save output
        for i in range(len(images)):
            print("Saving image...")
            image = images[i]
            image_metadata = PngInfo()
            image_metadata.add_text("Model", str(self._model))
            image_metadata.add_text("Prompt", str(self._prompt))
            image_metadata.add_text("Negative Prompt", str(self._negative_prompt))
            image_metadata.add_text("Guidance Scale", str(self._guidance_scale))
            image_metadata.add_text("Inference Step Count", str(self._inference_step_count))
            image.save(f"output/image_{i}.png", pnginfo=image_metadata)
            print("Done saving.")
