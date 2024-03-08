# Continuous Level Generation Using Conditional Diffusion Models

Reference: [Project](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

## Brief
This is a research project to generate levels for Super Mario Bros using diffusion model.

### Training/Resume Training
Set `resume_state` of configure file to the directory of previous checkpoint. Take the following as an example, this directory contains training states and saved model:

```yaml
"path": { //set every part file path
	"resume_state": "experiments/inpainting_celebahq_220426_150122/checkpoint/100" 
},
```
2. Set your network label in `load_everything` function of `model.py`, default is **Network**. Follow the tutorial settings, the optimizers and models will be loaded from 100.state and 100_Network.pth respectively.

3. Run the script:

```python
python3 run.py
```

4. Result
   
<img width="974" alt="res_overall_3" src="https://github.com/devshaww/DDPMLevelGeneration/assets/22312553/41b576ec-d33c-4294-8616-930b7c9154a8">
<img width="500" alt="res_overall_4" src="https://github.com/devshaww/DDPMLevelGeneration/assets/22312553/f7e6bef9-ae83-472d-afe4-8c181b5ab6bf">


## Acknowledge
This work is based on the following theoretical works:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

and we are benefiting a lot from the following projects:
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [LouisRouss/Diffusion-Based-Model-for-Colorization](https://github.com/LouisRouss/Diffusion-Based-Model-for-Colorization)
