import sys, traceback

from PySide6.QtCore import QRunnable, Slot, QThreadPool

from workers.worker_signals import WorkerSignals
from image import DSImage

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import  StableDiffusionUpscalePipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler

class UpscaleWorker(QRunnable):
    def __init__(self, image, scheduler, prompt, negative_prompt, seed, guidance_scale, inference_step_count):
        super(UpscaleWorker, self).__init__()

        #Public
        self.signals = WorkerSignals()

        #Private
        self._image = image
        self._model = "stabilityai/stable-diffusion-x4-upscaler"
        self._scheduler = scheduler
        self._prompt = prompt
        self._negative_prompt = negative_prompt

        if seed.isnumeric():
            self._seed = int(seed)
        elif "0x" in seed:
            self._seed = int(seed, 16)
        else:
            self._seed = None

        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count

    @Slot()
    def run(self):
        try:
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
                print(f"Scheduler not found! Defaulting to Euler Ancestral")
                pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

            #Send to GPU
            pipeline = pipeline.to("cuda")

            #Start generation
            print("Generating...")
            images = pipeline(
                image = self._image,
                prompt = self._prompt,
                negative_prompt = self._negative_prompt,
                guidance_scale = self._guidance_scale,
                num_inference_steps = self._inference_step_count,
                ).images[0]
            print("Done generating.")

            #Construct output objects
            output_images = []
            for i in range(len(images)):
                output_image = DSImage(
                    image = images[i],
                    model = self._model,
                    scheduler = self._scheduler,
                    prompt = self._prompt,
                    negative_prompt = self._negative_prompt,
                    seed = self._seed,
                    guidance_scale = self._guidance_scale,
                    inference_step_count = self._inference_step_count
                )
                output_images.append(output_image)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(output_images)
        finally:
            self.signals.finished.emit()
