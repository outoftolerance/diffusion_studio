import sys

from PySide6.QtCore import QObject

from PIL import Image
from PIL.PngImagePlugin import PngInfo

class DiffusionImage(QObject):
    @property
    def id(self):
        return self._id

    @property
    def image(self):
        return self._image

    @property
    def model(self):
        return self._model

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def prompt(self):
        return self._prompt

    @property
    def negative_prompt(self):
        return self._negative_prompt

    @property
    def seed(self):
        return self._seed

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def inference_step_count(self):
        return self._inference_step_count

    def __init__(self, image, model, scheduler, prompt, negative_prompt, seed, guidance_scale, inference_step_count):
        super(DiffusionImage, self).__init__()
        self._id = None
        self._image = image
        self._model = model
        self._scheduler = scheduler
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._seed = seed
        self._guidance_scale = guidance_scale
        self._inference_step_count = inference_step_count

    def set_id(self, new_id):
        self._id = new_id

    def save(self, output_directory):
        print(f"Saving image {output_directory}/image_{self._id}.png")

        image_metadata = PngInfo()
        image_metadata.add_text("Model", str(self._model))
        image_metadata.add_text("Scheduler", str(self._scheduler))
        image_metadata.add_text("Prompt", str(self._prompt))
        image_metadata.add_text("Negative Prompt", str(self._negative_prompt))
        image_metadata.add_text("Seed", str(hex(self._seed)))
        image_metadata.add_text("Guidance Scale", str(self._guidance_scale))
        image_metadata.add_text("Inference Step Count", str(self._inference_step_count))

        self._image.save(f"{output_directory}/image_{self._id}.png", pnginfo=image_metadata)
