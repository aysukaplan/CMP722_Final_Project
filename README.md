# CMP722_Final_Project
Entropy-Driven Test-Time Adaptation with Diffusion Models for Robust Semantic Segmentation in Adverse Weather

**Course:** CMP722: Advanced Computer Vision (Hacettepe University)
**Author:** Aysu Aylin Kaplan

## 📌 Project Overview
This project implements a **Test-Time Adaptation (TTA)** system to address domain shifts in semantic segmentation for autonomous driving. It utilizes a pre-trained diffusion model as a generative prior to "clean" adverse weather images (fog, rain, snow) at inference time, guided by the entropy of a segmentation model.

### Key Features
* **Stateless Adaptation:** No model weights are updated. Gradients are computed w.r.t pixel values, avoiding catastrophic forgetting.
* **Generative Prior:** Uses Stable Diffusion v1.5 to reconstruct clean images from adverse conditions.
* **Entropy Guidance:** The segmentation model steers the diffusion process to minimize prediction uncertainty.

## ⚙️ Methodology

### Architecture
The pipeline consists of two **frozen** models interacting at inference time:
1.  **Segmentation Model ($f_{seg}$):** PIDNet or SegFormer (Pre-trained on Cityscapes).
2.  **Diffusion Model ($f_{diff}$):** Stable Diffusion v1.5 (Latent Diffusion).

### TTA Guidance Loop
For an incoming adverse image $x_{adv}$, the process follows a reverse diffusion loop ($t = N \rightarrow 0$):

1.  **Prediction:** The diffusion model predicts a clean latent $z_0^{\text{pred}}$, decoded to $x_0^{\text{pred}}$.
2.  **Segmentation:** The frozen segmentation model predicts logits: $logits = f_{seg}(x_0^{\text{pred}})$.
3.  **Entropy Loss:** Calculate the mean entropy of the prediction
4.  **Gradient Computation:** Compute gradient w.r.t the clean latent:
    $$\mathbf{g} = \nabla_{z_0^{\text{pred}}} \mathcal{L}_{\text{entropy}}$$
5.  **Steering:** Update the noisy latent $z_t$ using a time-dependent schedule $s_t$:
    $$z_t^{\text{guided}} = z_t - s_t \cdot \mathbf{g}$$

## 📊 Experiment Setup

### Dataset Configuration
| Role | Dataset | Condition | Usage |
| :--- | :--- | :--- | :--- |
| **Source Domain** | Cityscapes | Clean Weather | Pre-trained $f_{seg}$ |
| **Target Domain** | ACDC | Fog, Night, Rain, Snow | Test |


## 🚀 Installation & Usage

### Prerequisites
* Anaconda or Miniconda
* Git

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aysukaplan/CMP722_Final_Project.git](https://github.com/aysukaplan/CMP722_Final_Project.git)
    cd CMP722_Final_Project
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    conda activate segtta
    ```
3. ACDC Dataset can be downloaded from the following link: https://acdc.vision.ee.ethz.ch/
4. PIDNet-L weights can be downloaded from the following link: https://github.com/XuJiacong/PIDNet  
   Place the weights(cityscapes) under pidnet/pretrained_models/cityscapes/
### Running Inference
To run the adaptation pipeline on the ACDC dataset:

```bash
# Example usage for pidnet
python /pidnet/TTA/tta_per_img_full_scalar_entropy.py --guidance_scale 1.0
```

```bash
# Example usage for segformer-b5
python /Segformer/tta_per_img_full_scalar_entropy.py --guidance_scale 1.0
