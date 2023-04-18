import sys, traceback

from PySide6.QtCore import QRunnable, Slot, QThreadPool

from workers.worker_signals import WorkerSignals
from image import DSImage

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import  StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler

class RemixWorker(QRunnable):
    def __init__(self, image, model, scheduler, prompt, negative_prompt, seed, seed_lock, width, height, noise_strength, guidance_scale, inference_step_count, image_count=1):
        super(RemixWorker, self).__init__()

        #Public
        self.signals = WorkerSignals()

        #Private
        self._image = image
        self._model = model
        self._scheduler = scheduler
        self._prompt = prompt
        self._negative_prompt = negative_prompt

        if seed.isnumeric():
            self._seed = int(seed)
        elif "0x" in seed:
            self._seed = int(seed, 16)
        else:
            self._seed = None
            
        self._seed_lock = seed_lock
        self._width = width
        self._height = height
        self._noise_strength = noise_strength
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count
        self._image_count = image_count

    @Slot()
    def run(self):
        try:
            #Setup optimizers
            torch.backends.cudnn.benchmark = True

            #Setup pipeline
            pipeline =  StableDiffusionImg2ImgPipeline.from_pretrained(
                self._model, 
                torch_dtype=torch.float16
            )

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

            #Slicing for memory optimisation
            pipeline.enable_attention_slicing()

            #Setup the seed
            generators = []

            #Choose to use the input seed or a random one
            if not self._seed == None:
                seed = self._seed
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
                image = self._image,
                prompt = self._prompt,
                negative_prompt = self._negative_prompt,
                generator = generators,
                strength = self._noise_strength,
                guidance_scale = self._guidance_scale,
                num_inference_steps = self._inference_step_count,
                num_images_per_prompt = self._image_count,
                ).images
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
                    seed = generators[i].initial_seed(),
                    guidance_scale = self._guidance_scale,
                    noise_strength = self._noise_strength,
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
