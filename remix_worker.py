from PySide6.QtCore import QRunnable, Slot, QThreadPool

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

class RemixWorker(QRunnable):
    def __init__(self, image, model, prompt, negative_prompt, noise_strength, guidance_scale, inference_step_count):
        super(RemixWorker, self).__init__()
        self._image = image.resize((512, 512))
        self._model = model
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._noise_strength = noise_strength
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count

    @Slot()
    def run(self):
        #Setup pipeline
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(self._model, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")

        #Start generation
        print("Generating...")
        image = pipeline(
            image = self._image,
            prompt = self._prompt,
            negative_prompt = self._negative_prompt,
            strength = self._noise_strength,
            guidance_scale = self._guidance_scale,
            num_inference_steps = self._inference_step_count,
            ).images[0]
        print("Done generating.")

        #Save output
        print("Saving image...")
        image_metadata = PngInfo()
        image_metadata.add_text("Model", str(self._model))
        image_metadata.add_text("Prompt", str(self._prompt))
        image_metadata.add_text("Negative Prompt", str(self._negative_prompt))
        image_metadata.add_text("Noise Scale", str(self._noise_strength))
        image_metadata.add_text("Guidance Scale", str(self._guidance_scale))
        image_metadata.add_text("Inference Step Count", str(self._inference_step_count))
        image.save(f"image_dream_remixed.png", pnginfo=image_metadata)
        print("Done saving.")
