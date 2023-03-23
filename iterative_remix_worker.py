from PySide6.QtCore import QRunnable, Slot, QThreadPool

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

class IterativeRemixWorker(QRunnable):
    def __init__(self, image, model, prompt, negative_prompt, noise_strength, guidance_scale, inference_step_count, image_count=1, iterations=1):
        super(IterativeRemixWorker, self).__init__()
        self._image = image.resize((512, 512))
        self._model = model
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._noise_strength = noise_strength
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count
        self._image_count = image_count
        self._iterations = iterations

    @Slot()
    def run(self):
        #Setup pipeline
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(self._model, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")

        #Init the input images storage
        input_images = [self._image] * self._image_count

        #Start generation
        for iteration in range(0, self._iterations):
            print("Generating...")
            images = pipeline(
                image = input_images,
                prompt = [self._prompt] * self._image_count,
                negative_prompt = [self._negative_prompt] * self._image_count,
                strength = self._noise_strength,
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
                image.save(f"output/image_remix_{i}_{iteration}.png", pnginfo=image_metadata)
                print("Done saving.")

            #Copy output to input for next loop
            input_images = images