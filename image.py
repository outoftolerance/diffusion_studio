import sys

from PySide6 import QtGui
from PySide6.QtCore import QObject

from PIL import Image
from PIL.ImageQt import ImageQt
from PIL.PngImagePlugin import PngInfo

class DSImage(QObject):
    @property
    def id(self):
        return self._id

    @property
    def image(self):
        return self._image
    
    @property
    def pixmap(self):
        qt_image = ImageQt(self._image)
        return QtGui.QPixmap.fromImage(qt_image)
    
    @property
    def size(self):
        return self._image.size

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
    def noise_strength(self):
        return self._noise_strength

    @property
    def inference_step_count(self):
        return self._inference_step_count

    def __init__(self, id=None, image=None, model=None, scheduler=None, prompt="", negative_prompt="", seed=0, guidance_scale=0.0, noise_strength=0.0, inference_step_count=0):
        super(DSImage, self).__init__()
        self._id = id
        self._image = image
        self._model = model
        self._scheduler = scheduler
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._seed = seed
        self._guidance_scale = guidance_scale
        self._noise_strength = noise_strength
        self._inference_step_count = inference_step_count

    def set_id(self, new_id):
        self._id = new_id
        return True
    
    def open(self, filepath):
        self._image = Image.open(filepath)

        width = self._image.size[0]
        height = self._image.size[1]

        if "Model" in self._image.info:
            self._model = self._image.info["Model"]

        if "Scheduler" in self._image.info:
            self._scheduler = self._image.info["Scheduler"]

        if "Prompt" in self._image.info:
            self._prompt = self._image.info["Prompt"]

        if "Negative Prompt" in self._image.info:
            self._negative_prompt = self._image.info["Negative Prompt"]

        if "Seed" in self._image.info:
            seed = self._image.info["Seed"]
            if seed.isnumeric():
                self._seed = int(seed)
            elif "0x" in seed:
                self._seed = int(seed, 16)
            else:
                self._seed = None

        if "Guidance Scale" in self._image.info:
            self._guidance_scale = float(self._image.info["Guidance Scale"])

        if "Noise Strength" in self._image.info:
            self._noise_strength = float(self._image.info["Noise Strength"])

        if "Inference Step Count" in self._image.info:
            self._inference_step_count = int(self._image.info["Inference Step Count"])

        return True

    def save(self, directory, filename=None):
        if not filename:
            filename = "image_" + str(self._id)
        
        print(f"Saving image {directory}/{filename}")

        image_metadata = PngInfo()
        image_metadata.add_text("Model", str(self._model))
        image_metadata.add_text("Scheduler", str(self._scheduler))
        image_metadata.add_text("Prompt", str(self._prompt))
        image_metadata.add_text("Negative Prompt", str(self._negative_prompt))
        image_metadata.add_text("Seed", str(hex(self._seed)))
        image_metadata.add_text("Guidance Scale", str(self._guidance_scale))
        image_metadata.add_text("Inference Step Count", str(self._inference_step_count))

        self._image.save(f"{directory}/{filename}.png", pnginfo=image_metadata)

        return True
