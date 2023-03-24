import sys, time

from PIL import Image

from PySide6.QtCore import Qt, QRunnable, Slot, QThreadPool 
from PySide6.QtWidgets import *

from diffusion_worker import DiffusionWorker
from remix_worker import RemixWorker
from iterative_remix_worker import IterativeRemixWorker
from upscale_worker import UpscaleWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Diffusion GUI")

        self.diffusion_models = [
            {
                "name": "Stable Diffusion V1.5",
                "repo": "runwayml/stable-diffusion-v1-5",
            },
            {
                "name": "Stable Diffusion V2.1",
                "repo": "stabilityai/stable-diffusion-2-1",
            },
            {
                "name": "Openjourney",
                "repo": "prompthero/openjourney",
            },
            {
                "name": "Openjourney V4.0",
                "repo": "prompthero/openjourney-v4",
            },
            {
                "name": "N6AI Graphic Art",
                "repo": "n6ai/graphic-art",
            },
            {
                "name": "Lykon DreamShaper",
                "repo": "Lykon/DreamShaper",
            },
            {
                "name": "Darkstorm Protogen V5.8",
                "repo": "darkstorm2150/Protogen_x5.8_Official_Release",
            },
            {
                "name": "SG161222 Realistic Vision V1.4",
                "repo": "SG161222/Realistic_Vision_V1.4",
            },
        ]

        self.schedulers = [
            {
                "name": "Euler Ancestral",
                "class": "EulerAncestralDiscreteScheduler",
            },
            {
                "name": "Euler",
                "class": "EulerDiscreteScheduler",
            },
            {
                "name": "DDIM",
                "class": "DDIMScheduler",
            },
            {
                "name": "DDPM",
                "class": "DDPMScheduler",
            },
            {
                "name": "Multistep DPM Solver",
                "class": "DPMSolverMultistepScheduler",
            },
            {
                "name": "Singlestep DPM Solver",
                "class": "DPMSolverSinglestepScheduler",
            },
        ]

        self.threadpool = QThreadPool()

        ### IMAGE LOADER
        ### -------------------------------------------------------------------------
        self.label_load_image = QLabel("Load Image:")

        self.lineedit_load_image = QLineEdit()
        self.button_load_image = QPushButton("Load Image")
        self.button_load_image.clicked.connect(self.load_image)

        ### DIFFUSION MODEL
        ### -------------------------------------------------------------------------
        self.label_diffusion_model = QLabel("Diffusion Model:")

        self.dropdown_diffusion_model = QComboBox()
        for model in self.diffusion_models:
            self.dropdown_diffusion_model.addItem(model["name"])

        ### SCHEDULER
        ### -------------------------------------------------------------------------
        self.label_scheduler = QLabel("Scheduler:")

        self.dropdown_scheduler = QComboBox()
        for scheduler in self.schedulers:
            self.dropdown_scheduler.addItem(scheduler["name"])

        self.label_prompts = QLabel("Prompts:")

        self.textarea_prompt = QPlainTextEdit()
        self.textarea_prompt.setPlaceholderText("Positive Prompt")
        self.textarea_negative_prompt =QPlainTextEdit()
        self.textarea_negative_prompt.setPlaceholderText("Negative Prompt")

        ### SEED
        ### -------------------------------------------------------------------------
        self.label_seed = QLabel("Seed:")

        self.lineedit_seed = QLineEdit()

        self.widget_seed_lock = QWidget()
        self.layout_seed_lock = QHBoxLayout()
        self.layout_seed_lock.setContentsMargins(0,0,0,0)

        self.label_seed_lock = QLabel("Lock Seed:")
        self.label_seed_lock.setAlignment(Qt.AlignLeft)
        self.checkbox_seed_lock = QCheckBox()

        self.layout_seed_lock.addWidget(self.label_seed_lock)
        self.layout_seed_lock.addStretch()
        self.layout_seed_lock.addWidget(self.checkbox_seed_lock)

        self.widget_seed_lock.setLayout(self.layout_seed_lock)

        ### IMAGE SIZE
        ### -------------------------------------------------------------------------
        self.label_output_image_size = QLabel("Image Size:")

        self.widget_output_image_size = QWidget()
        self.layout_output_image_size = QHBoxLayout()
        self.layout_output_image_size.setContentsMargins(0,0,0,0)

        self.label_output_image_width = QLabel("Width:")
        self.spinbox_output_image_width = QSpinBox()
        self.spinbox_output_image_width.setMinimum(128)
        self.spinbox_output_image_width.setMaximum(2048)
        self.spinbox_output_image_width.setSingleStep(8)
        self.spinbox_output_image_width.setValue(512)

        self.label_output_image_height = QLabel("Height:")
        self.spinbox_output_image_height = QSpinBox()
        self.spinbox_output_image_height.setMinimum(128)
        self.spinbox_output_image_height.setMaximum(2048)
        self.spinbox_output_image_height.setSingleStep(8)
        self.spinbox_output_image_height.setValue(512)

        self.layout_output_image_size.addWidget(self.label_output_image_width)
        self.layout_output_image_size.addWidget(self.spinbox_output_image_width)
        self.layout_output_image_size.addWidget(self.label_output_image_height)
        self.layout_output_image_size.addWidget(self.spinbox_output_image_height)
        self.widget_output_image_size.setLayout(self.layout_output_image_size)

        ### REMIX NOISE STRENGTH
        ### -------------------------------------------------------------------------
        self.widget_noise_strength = QWidget()
        self.layout_noise_strength = QHBoxLayout()
        self.layout_noise_strength.setContentsMargins(0,0,0,0)

        self.label_noise_strength = QLabel("Remix Noise Strength:")

        self.slider_noise_strength = QSlider(Qt.Horizontal)
        self.slider_noise_strength.setMinimum(1)
        self.slider_noise_strength.setMaximum(100)
        self.slider_noise_strength.setTickInterval(10)
        self.slider_noise_strength.setSingleStep(10)
        self.slider_noise_strength.setTickPosition(QSlider.TicksBelow)
        self.slider_noise_strength.setValue(40)
        self.slider_noise_strength.valueChanged.connect(self.ui_slider_update)

        self.lineedit_noise_strength = QLineEdit()
        self.lineedit_noise_strength.setText("40")
        self.lineedit_noise_strength.setMaximumWidth(25)

        self.layout_noise_strength.addWidget(self.slider_noise_strength)
        self.layout_noise_strength.addWidget(self.lineedit_noise_strength)
        self.widget_noise_strength.setLayout(self.layout_noise_strength)

        ### GUIDANCE SCALE
        ### -------------------------------------------------------------------------
        self.widget_guidance_scale = QWidget()
        self.layout_guidance_scale = QHBoxLayout()
        self.layout_guidance_scale.setContentsMargins(0,0,0,0)

        self.label_guidance_scale = QLabel("Guidance Scale:")

        self.slider_guidance_scale = QSlider(Qt.Horizontal)
        self.slider_guidance_scale.setMinimum(1)
        self.slider_guidance_scale.setMaximum(100)
        self.slider_guidance_scale.setTickInterval(10)
        self.slider_guidance_scale.setSingleStep(10)
        self.slider_guidance_scale.setTickPosition(QSlider.TicksBelow)
        self.slider_guidance_scale.setValue(70)
        self.slider_guidance_scale.valueChanged.connect(self.ui_slider_update)

        self.lineedit_guidance_scale = QLineEdit()
        self.lineedit_guidance_scale.setText("70")
        self.lineedit_guidance_scale.setMaximumWidth(25)

        self.layout_guidance_scale.addWidget(self.slider_guidance_scale)
        self.layout_guidance_scale.addWidget(self.lineedit_guidance_scale)
        self.widget_guidance_scale.setLayout(self.layout_guidance_scale)

        ### INFERENCE STEP COUNT
        ### -------------------------------------------------------------------------
        self.widget_inference_step_count = QWidget()
        self.layout_inference_step_count = QHBoxLayout()
        self.layout_inference_step_count.setContentsMargins(0,0,0,0)

        self.label_inference_step_count = QLabel("Inference Step Count:")

        self.slider_inference_step_count = QSlider(Qt.Horizontal)
        self.slider_inference_step_count.setMinimum(8)
        self.slider_inference_step_count.setMaximum(128)
        self.slider_inference_step_count.setTickInterval(8)
        self.slider_inference_step_count.setSingleStep(1)
        self.slider_inference_step_count.setTickPosition(QSlider.TicksBelow)
        self.slider_inference_step_count.setValue(48)
        self.slider_inference_step_count.valueChanged.connect(self.ui_slider_update)

        self.lineedit_inference_step_count = QLineEdit()
        self.lineedit_inference_step_count.setText("48")
        self.lineedit_inference_step_count.setMaximumWidth(25)

        self.layout_inference_step_count.addWidget(self.slider_inference_step_count)
        self.layout_inference_step_count.addWidget(self.lineedit_inference_step_count)
        self.widget_inference_step_count.setLayout(self.layout_inference_step_count)

        ### OUTPUT IMAGE COUNT
        ### -------------------------------------------------------------------------
        self.label_output_image_count = QLabel("Batch Size:")

        self.spinbox_output_image_count = QSpinBox()
        self.spinbox_output_image_count.setMinimum(1)
        self.spinbox_output_image_count.setMaximum(32)
        self.spinbox_output_image_count.setSingleStep(1)
        self.spinbox_output_image_count.setValue(4)

        ### ITERATIVE REMIX LENGTH
        ### -------------------------------------------------------------------------
        self.label_iterative_remix_count = QLabel("Iterative Remix Count:")

        self.spinbox_iterative_remix_count = QSpinBox()
        self.spinbox_iterative_remix_count.setMinimum(1)
        self.spinbox_iterative_remix_count.setMaximum(64)
        self.spinbox_iterative_remix_count.setSingleStep(1)
        self.spinbox_iterative_remix_count.setValue(4)

        ### BUTTONS
        ### -------------------------------------------------------------------------
        self.button_diffuse = QPushButton("Generate")
        self.button_diffuse.clicked.connect(self.execute_diffusion)

        self.button_remix = QPushButton("Remix")
        self.button_remix.clicked.connect(self.execute_remix)

        self.button_iterative_remix = QPushButton("Iterative Remix")
        self.button_iterative_remix.clicked.connect(self.execute_iterative_remix)

        self.button_upscale = QPushButton("Upscale")
        self.button_upscale.clicked.connect(self.execute_upscale)

        ### CONSTRUCT MAIN LAYOUT
        ### -------------------------------------------------------------------------
        self.layout_main_window = QVBoxLayout()
        self.widget_main_window = QWidget()

        self.layout_main_window.addWidget(self.label_load_image)
        self.layout_main_window.addWidget(self.lineedit_load_image)
        self.layout_main_window.addWidget(self.button_load_image)
        self.layout_main_window.addWidget(self.label_diffusion_model)
        self.layout_main_window.addWidget(self.dropdown_diffusion_model)
        self.layout_main_window.addWidget(self.label_scheduler)
        self.layout_main_window.addWidget(self.dropdown_scheduler)
        self.layout_main_window.addWidget(self.label_prompts)
        self.layout_main_window.addWidget(self.textarea_prompt)
        self.layout_main_window.addWidget(self.textarea_negative_prompt)
        self.layout_main_window.addWidget(self.label_seed)
        self.layout_main_window.addWidget(self.lineedit_seed)
        self.layout_main_window.addWidget(self.widget_seed_lock)
        self.layout_main_window.addWidget(self.label_output_image_size)
        self.layout_main_window.addWidget(self.widget_output_image_size)
        self.layout_main_window.addWidget(self.label_output_image_count)
        self.layout_main_window.addWidget(self.spinbox_output_image_count)
        self.layout_main_window.addWidget(self.label_guidance_scale)
        self.layout_main_window.addWidget(self.widget_guidance_scale)
        self.layout_main_window.addWidget(self.label_inference_step_count)
        self.layout_main_window.addWidget(self.widget_inference_step_count)
        self.layout_main_window.addWidget(self.button_diffuse)
        self.layout_main_window.addWidget(self.label_noise_strength)
        self.layout_main_window.addWidget(self.widget_noise_strength)
        self.layout_main_window.addWidget(self.button_remix)
        self.layout_main_window.addWidget(self.label_iterative_remix_count)
        self.layout_main_window.addWidget(self.spinbox_iterative_remix_count)
        self.layout_main_window.addWidget(self.button_iterative_remix)
        self.layout_main_window.addWidget(self.button_upscale)

        self.widget_main_window.setLayout(self.layout_main_window)

        self.setCentralWidget(self.widget_main_window)

    def ui_slider_update(self):
        value = self.slider_noise_strength.value()
        self.lineedit_noise_strength.setText(str(value))

        value = self.slider_guidance_scale.value()
        self.lineedit_guidance_scale.setText(str(value))

        value = self.slider_inference_step_count.value()
        self.lineedit_inference_step_count.setText(str(value))

    def load_image(self):
        #Get file from dialog
        filename = QFileDialog().getOpenFileName()

        #Check for cancellation
        if len(filename[0]) == 0:
            return

        self.lineedit_load_image.setText(filename[0])

        model = None
        prompt = None
        negative_prompt = None
        noise_strength = None
        guidance_scale = None
        inference_step_count = None

        image = Image.open(filename[0])

        if "Model" in image.info:
            loaded_model = image.info["Model"]
            for model in self.diffusion_models:
                if model["repo"] == loaded_model:
                    self.dropdown_diffusion_model.setCurrentText(model["name"])

        if "Scheduler" in image.info:
            loaded_scheduler = image.info["Scheduler"]
            for scheduler in self.schedulers:
                if scheduler["class"] == loaded_model:
                    self.dropdown_scheduler.setCurrentText(scheduler["name"])

        if "Prompt" in image.info:
            prompt = image.info["Prompt"]
            self.textarea_prompt.setPlainText(prompt)

        if "Negative Prompt" in image.info:
            negative_prompt = image.info["Negative Prompt"]
            self.textarea_negative_prompt.setPlainText(negative_prompt)

        if "Seed" in image.info:
            seed = image.info["Seed"]
            self.lineedit_seed.setText(seed)

        width = image.size[0]
        self.spinbox_output_image_width.setValue(int(width))

        height = image.size[1]
        self.spinbox_output_image_height.setValue(int(height))

        if "Noise Strength" in image.info:
            noise_strength = image.info["Noise Strength"]
            noise_strength_int = int(round(float(noise_strength) * 100, 0))
            self.slider_noise_strength.setValue(noise_strength_int)

        if "Guidance Scale" in image.info:
            guidance_scale = image.info["Guidance Scale"]
            guidance_scale_int = int(round(float(guidance_scale) * 100, 0))
            self.slider_guidance_scale.setValue(guidance_scale_int)

        if "Inference Step Count" in image.info:
            inference_step_count = image.info["Inference Step Count"]
            inference_step_count_int = int(inference_step_count)
            self.slider_inference_step_count.setValue(inference_step_count_int)

        return

    def get_diffusion_model_from_name(self, name):
        for model in self.diffusion_models:
            if model["name"] == self.dropdown_diffusion_model.currentText():
                return model["repo"]

        return None

    def get_scheduler_from_name(self, name):
        for scheduler in self.schedulers:
            if scheduler["name"] == self.dropdown_scheduler.currentText():
                return scheduler["class"]

        return None

    def execute_diffusion(self):
        #Configure worker
        diffusion_worker = DiffusionWorker(
            model = self.get_diffusion_model_from_name(self.dropdown_diffusion_model.currentText()),
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText()),
            prompt = self.textarea_prompt.toPlainText(),
            negative_prompt = self.textarea_negative_prompt.toPlainText(),
            seed = self.lineedit_seed.text(),
            seed_lock = self.checkbox_seed_lock.isChecked(),
            width = int(self.spinbox_output_image_width.value()),
            height = int(self.spinbox_output_image_height.value()),
            guidance_scale = round(self.slider_guidance_scale.value()/100.0, 2),
            inference_step_count = self.slider_inference_step_count.value(),
            image_count = self.spinbox_output_image_count.value(),
        )

        #Run worker
        self.threadpool.start(diffusion_worker)

    def execute_remix(self):
        #Get the image
        image = Image.open(self.lineedit_load_image.text()).convert("RGB")

        #Configure worker
        remix_worker = RemixWorker(
            image = image,
            model = self.get_diffusion_model_from_name(self.dropdown_diffusion_model.currentText()),
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText()),
            prompt = self.textarea_prompt.toPlainText(),
            negative_prompt = self.textarea_negative_prompt.toPlainText(),
            seed = self.lineedit_seed.text(),
            seed_lock = self.checkbox_seed_lock.isChecked(),
            width = self.spinbox_output_image_width.value(),
            height = self.spinbox_output_image_height.value(),
            noise_strength = round(self.slider_noise_strength.value()/100.0, 2),
            guidance_scale = round(self.slider_guidance_scale.value()/100.0, 2),
            inference_step_count = self.slider_inference_step_count.value(),
            image_count = self.spinbox_output_image_count.value(),
        )

        #Run worker
        self.threadpool.start(remix_worker)

    def execute_iterative_remix(self):
        #Get the image
        image = Image.open(self.lineedit_load_image.text()).convert("RGB")

        #Configure worker
        iterative_remix_worker = IterativeRemixWorker(
            image = image,
            model = self.get_diffusion_model_from_name(self.dropdown_diffusion_model.currentText()),
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText()),
            prompt = self.textarea_prompt.toPlainText(),
            negative_prompt = self.textarea_negative_prompt.toPlainText(),
            seed = self.lineedit_seed.text(),
            seed_lock = self.checkbox_seed_lock.isChecked(),
            noise_strength = round(self.slider_noise_strength.value()/100.0, 2),
            guidance_scale = round(self.slider_guidance_scale.value()/100.0, 2),
            inference_step_count = self.slider_inference_step_count.value(),
            image_count = self.spinbox_output_image_count.value(),
            iterations = self.spinbox_iterative_remix_count.value(),
        )

        #Run worker
        self.threadpool.start(iterative_remix_worker)

    def execute_upscale(self):
        #Get the image
        image = Image.open("image_dream.png").convert("RGB")

        #Configure worker
        upscale_worker = UpscaleWorker(
            image = image,
            prompt = self.textarea_prompt.toPlainText(),
            negative_prompt = self.textarea_negative_prompt.toPlainText(),
            seed = self.lineedit_seed.text(),
            width = int(self.spinbox_output_image_width.value()),
            height = int(self.spinbox_output_image_height.value()),
            guidance_scale = round(self.slider_guidance_scale.value()/100.0, 2),
            inference_step_count = self.slider_inference_step_count.value(),
        )

        #Run worker
        self.threadpool.start(upscale_worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window=MainWindow()
    window.show()

    sys.exit(app.exec())