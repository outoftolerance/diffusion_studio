# Diffusion GUI
Diffusion GUI is a simple user interface that wraps the diffusers library and allows you to more easily run diffusion models without needing to install more complex systems such as AUTOMATIC1111. Diffusion GUI supports basic image generation, image-2-image remixing, and upscaling (if you have a large enough GPU). The user can specify a prompt, negative prompt, the guidance scale, number of iterations, and the noise scale for remixes.

The application will save all user inputs as metadata in the output PNG files, this allows the user to load a previously generated or processed image and remix it with the same prompts without needing to store/remember the prompts elsewhere. This dramatically simplifies output file management.

Multiple diffision models are supported from the HuggingFace repository, technically you can use any you like, I've just selected some neat looking ones after browsing around for a few minutes.

# Installation
Firstly, you need to install CUDA 11.8 and cuDNN for CUDA 11.8. Head to Nvidia's website to get these packages.

Once that's done, just install with pip and you're good to go.

```
pip install -r requirements.txt
python diffusion_gui.py
```

# Functionality

## Model Setup

- **Diffusion Model:** Dropdown selection box of some popular diffusion models from the HuggingFace repository.
- **Scheduler:** Dropdown selection box of some popular diffusion schedulers. This is a subset of what the diffusers library supports but is what most people seem to be using most of the time.

## Prompts

- **Prompt:** What you want.
- **Negative Prompt:** What you don't want.

## Diffusion Settings

- **Seed:** The input seed, integer in ASCII. Doesn't support hex at this time. Will auto-increment the seed by 1 for each of the output images.
- **Lock Seed:** If checked, will lock the seed for each of the output images. Implemented for future prompt mixing with knowng good seed.
- **Guidance Scale:** How close you want the model to stick to your prompt, if it's generating output that you like then lower this value to stick closely to it when remixing.
- **Inference Step Count:** The number of inference steps to perform during *Generation* or *Remixing*
- **Output Image Count:** Sets the number of output images from all actions except upscaling. This is limited from 1-4 to prevent processing times shooting off to the moon with high output image counts.

## Remix Settings

The following settings only apply to remixes and not to new image generation.

- **Noise Strength:** Sets the strength of the noise applied to the input image when *Remixing*. The more noise applied, the more the output will diverge from the input.
- **Iterative Remix Count:** The number of iterations that you want to loop for when performing the *Iterative Remixing* action.

## Image Loading

Previously generated images, or any other input image, can be loaded by the app. If you run the *Remix* action the loaded image will be used as input to the image2image diffusion pipeline.

Loading an image that was previously generated by Diffusion GUI will automatically load the Diffusion Model, Prompt, Negative Prompt, Guidance Scale, Inference Step Count, and Noise Strength that were used to *Generate*/*Remix* the loaded image. These values are loaded from the PNG metadata that is embeded in the output images from Diffusion GUI.

## Actions

- **Generate:** Creates a brand new image or images (determined by the *Output Image Count* setting) from your model selection, prompts and other parameters.
- **Remix:** Remixes an image by taking the loaded input iamge and passing it to the selected model as an image2image input along with the prompts and oahter parameters.
- **Iterative Remix:** This action will remix the loaded input image, then load the remixed image back in for more remixing, and repeat this for a given number of iterations. This is cool if you want to make an animation/video of the AI changing the image slightly with each iteration. It works with the *Output Image Count* input too, so you can have up to 4 diverging loops going at once.
- **Upscale:** Do this to upscale an the loaded input image to a higher resolution. Supports the Stable Diffusion x4 model only.