import sys, time, glob

from PySide6.QtCore import Qt, QRunnable, Slot, QThreadPool 
from PySide6.QtWidgets import *

from PIL import Image
from natsort import os_sorted

from image import DSImage
from workers.diffusion_worker import DiffusionWorker
from workers.remix_worker import RemixWorker
from workers.iterative_remix_worker import IterativeRemixWorker
from workers.upscale_worker import UpscaleWorker

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
                "name": "Huen",
                "class": "HeunDiscreteScheduler",
            },
            {
                "name": "Linear Multistep (LMS)",
                "class": "LMSDiscreteScheduler",
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
        self.spinbox_output_image_width.setSingleStep(128)
        self.spinbox_output_image_width.setValue(512)

        self.label_output_image_height = QLabel("Height:")
        self.spinbox_output_image_height = QSpinBox()
        self.spinbox_output_image_height.setMinimum(128)
        self.spinbox_output_image_height.setMaximum(2048)
        self.spinbox_output_image_height.setSingleStep(128)
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

    ## UI FUNCS
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
            return False

        #Update UI with selected file
        self.lineedit_load_image.setText(filename[0])

        #Load
        image = DSImage()
        image.open(filename[0])

        #Update UI with file contents
        self.load_ui_from_image(image)

        return True
    
    ## LOADERS
    def load_ui_from_image(self, image):
        self.load_model_from_image(image)
        self.load_scheduler_from_image(image)
        self.load_prompt_from_image(image)
        self.load_negative_prompt_from_image(image)
        self.load_resolution_from_image(image)
        self.load_seed_from_image(image)
        self.load_guidance_scale_from_image(image)
        self.load_noise_strength_from_image(image)
        self.load_inference_step_count_from_image(image)

    def load_model_from_image(self, image):
        if len(image.model) > 0:
            model = self.get_diffusion_model_from_repo(image.model)
            if model:
                self.dropdown_scheduler.setCurrentText(model["name"])
                return True
        
        return False

    def load_scheduler_from_image(self, image):
        if len(image.scheduler) > 0:
            scheduler = self.get_scheduler_from_class(image.scheduler)
            if scheduler:
                self.dropdown_scheduler.setCurrentText(scheduler["name"])
                return True
        
        return False

    def load_prompt_from_image(self, image):
        if len(image.prompt) > 0:
            self.textarea_prompt.setPlainText(image.prompt)
            return True
        
        return False
    
    def load_negative_prompt_from_image(self, image):
        if len(image.negative_prompt) > 0:
            self.textarea_negative_prompt.setPlainText(image.negative_prompt)
            return True
        
        return False
    
    def load_resolution_from_image(self, image):
        width = image.size[0]
        self.spinbox_output_image_width.setValue(int(width))

        height = image.size[1]
        self.spinbox_output_image_height.setValue(int(height))

    def load_seed_from_image(self, image):
        if image.seed > 0:
            self.lineedit_seed.setText(str(image.seed))
            return True
        
        return False

    def load_guidance_scale_from_image(self, image):
        if image.guidance_scale > 0:
            guidance_scale_int = int(round(float(image.guidance_scale) * 100, 0))
            self.slider_guidance_scale.setValue(guidance_scale_int)
            return True

        return False

    def load_noise_strength_from_image(self, image):
        if image.noise_strength > 0:
            noise_strength_int = int(round(float(image.noise_strength) * 100, 0))
            self.slider_noise_strength.setValue(noise_strength_int)
            return True
        
        return False

    def load_inference_step_count_from_image(self, image):
        if image.inference_step_count > 0:
            inference_step_count_int = int(round(float(image.inference_step_count) * 100, 0))
            self.slider_inference_step_count.setValue(inference_step_count_int)
            return True
        
        return False

    ## UTILITIES
    def get_diffusion_model_from_name(self, name):
        for model in self.diffusion_models:
            if model["name"] == name:
                return model

        return None
    
    def get_diffusion_model_from_repo(self, repo):
        for model in self.diffusion_models:
            if model["repo"] == repo:
                return model

        return None

    def get_scheduler_from_name(self, name):
        for scheduler in self.schedulers:
            if scheduler["name"] == name:
                return scheduler

        return None
    
    def get_scheduler_from_class(self, class_name):
        for scheduler in self.schedulers:
            if scheduler["class"] == class_name:
                return scheduler

        return None
    
    def get_next_image_id(self):
        existing_images = glob.glob("./output/*.png")

        if len(existing_images) > 0:
            existing_images = os_sorted(existing_images, reverse=True)
            highest_existing_id = int(existing_images[0].split("_")[1].split(".")[0])
            next_id = highest_existing_id + 1
        else:
            next_id = 0

        return next_id

    ## DIFFUSION
    def execute_diffusion(self):
        #Configure worker
        diffusion_worker = DiffusionWorker(
            model = self.get_diffusion_model_from_name(self.dropdown_diffusion_model.currentText())["repo"],
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText())["class"],
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

        diffusion_worker.signals.progress.connect(self.on_worker_progress)
        diffusion_worker.signals.result.connect(self.on_worker_result)
        diffusion_worker.signals.finished.connect(self.on_worker_finished)
        diffusion_worker.signals.error.connect(self.on_worker_error)

        #Run worker
        self.threadpool.tryStart(diffusion_worker)

    def execute_remix(self):
        #Get the image
        image = Image.open(self.lineedit_load_image.text()).convert("RGB")

        #Configure worker
        remix_worker = RemixWorker(
            image = image,
            model = self.get_diffusion_model_from_name(self.dropdown_diffusion_model.currentText())["repo"],
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText())["class"],
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

        remix_worker.signals.progress.connect(self.on_worker_progress)
        remix_worker.signals.result.connect(self.on_worker_result)
        remix_worker.signals.finished.connect(self.on_worker_finished)
        remix_worker.signals.error.connect(self.on_worker_error)

        #Run worker
        self.threadpool.tryStart(remix_worker)

    def execute_iterative_remix(self):
        #Get the image
        image = Image.open(self.lineedit_load_image.text()).convert("RGB")

        #Configure worker
        iterative_remix_worker = IterativeRemixWorker(
            image = image,
            model = self.get_diffusion_model_from_name(self.dropdown_diffusion_model.currentText())["repo"],
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText())["class"],
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

        iterative_remix_worker.signals.progress.connect(self.on_worker_progress)
        iterative_remix_worker.signals.result.connect(self.on_worker_result)
        iterative_remix_worker.signals.finished.connect(self.on_worker_finished)
        iterative_remix_worker.signals.error.connect(self.on_worker_error)

        #Run worker
        self.threadpool.tryStart(iterative_remix_worker)

    def execute_upscale(self):
        #Get the image
        image = Image.open(self.lineedit_load_image.text()).convert("RGB")

        #Configure worker
        upscale_worker = UpscaleWorker(
            image = image,
            scheduler = self.get_scheduler_from_name(self.dropdown_scheduler.currentText())["class"],
            prompt = self.textarea_prompt.toPlainText(),
            negative_prompt = self.textarea_negative_prompt.toPlainText(),
            seed = self.lineedit_seed.text(),
            guidance_scale = round(self.slider_guidance_scale.value()/100.0, 2),
            inference_step_count = self.slider_inference_step_count.value(),
        )

        upscale_worker.signals.progress.connect(self.on_worker_progress)
        upscale_worker.signals.result.connect(self.on_worker_result)
        upscale_worker.signals.finished.connect(self.on_worker_finished)
        upscale_worker.signals.error.connect(self.on_worker_error)

        #Run worker
        self.threadpool.tryStart(upscale_worker)

    ## SIGNAL CALLBACKS
    def on_worker_progress(self, progression):
        print("Progressing...")

    def on_worker_result(self, images):
        print("Results!")
        next_id = self.get_next_image_id()

        for image in images:
            image.set_id(next_id)
            image.save("./output")
            next_id += 1

    def on_worker_finished(self):
        print("Finished!")

    def on_worker_error(self, error_info):
        print("Error!")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window=MainWindow()
    window.show()

    sys.exit(app.exec())