# Diffusion GUI
Diffusion GUI is a simple user interface that wraps the diffusers library and allows you to more easily run diffusion models without needing to install more complex systems such as AUTOMATIC1111. Diffusion GUI supports basic image generation, image-2-image remixing, and upscaling (if you have a large enough GPU). The user can specify a prompt, negative prompt, the guidance scale, number of iterations, and the noise scale for remixes.

The application will save all user inputs as metadata in the output PNG files, this allows the user to load a previously generated or processed image and remix it with the same prompts without needing to store/remember the prompts elsewhere. This dramatically simplifies output file management.

Multiple diffision models are supported from the HuggingFace repository, technically you can use any you like, I've just selected some neat looking ones after browsing around for a few minutes.

# Installation
Firstly, you need to install CUDA and cuDNN.

Once that's done, just install with pip and you're good to go.

```pip install -r requirements.txt```
```python diffusion_gui.py```