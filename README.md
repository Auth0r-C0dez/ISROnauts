
# Project Title

We aim to develop a deep generative model that forecasts short‑term (0–3 hrs) cloud motion from satellite imagery. Traditional optical‑flow and physics‑based methods struggle with complex, rapidly evolving cloud dynamics—particularly for severe weather nowcasting. Our objective is to harness conditional diffusion networks to learn spatio‑temporal patterns across multi‑spectral INSAT‑3DR/3DS frames and generate realistic future cloud frames for improved forecasting.
## Acknowledgements

We would like to express our sincere gratitude to the following organizations and tools that made this project possible:

#### ISRO / IMD –
 for providing publicly available multi-spectral INSAT satellite imagery, which served as the core dataset for this research.
#### Google Scholar/ScienceDirect – 
for assistance in research
#### Hugging Face – 
for open-source libraries and model hosting, particularly diffusers, which inspired key parts of the pipeline.

####PyTorch – 
for providing the deep learning framework used to build and train the UNet3D and diffusion models.

####video-diffusion-pytorch by lucidrains – 
for the initial implementation of 3D diffusion, which was adapted and extended for satellite cloud forecasting.

####Google Colab– 
for offering accessible GPU compute resources for model training and prototyping.
