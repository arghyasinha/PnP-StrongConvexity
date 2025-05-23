# 🧑‍💻 Code for the Paper

**On the Strong Convexity of PnP Regularization Using Linear Denoisers**  
*Arghya Sinha, Kunal N. Chaudhury*  
IEEE Signal Processing Letters, 2024, Vol. 31, pp. 2790–2794  
[DOI: 10.1109/LSP.2024.3475913](https://doi.org/10.1109/LSP.2024.3475913)

---

This repository contains code for the key experiments from our paper.

---

## 👥 Authors

- [Arghya Sinha](https://arghyasinha.github.io)
- [Kunal N. Chaudhury](https://sites.google.com/site/kunalnchaudhury/home)

---

## 🧪 Experiments

### 📈 Figure 1: Strong Convexity Index

- Estimates a lower bound for the strong convexity index of the objective function.
- Code: [`Figure_1_Strong_Convexity_index.py`](Figure_1_Strong_Convexity_index.py)
- **Arguments:**  
  - `forward_model`: `"inpainting"` or `"deblurring"`
  - `image_id`: Name of the image in the `images` folder.
  - `sigma`: Noise level (integer in `[0, 255]`)
- **Example usage:**
  ```bash
  python Figure_1_Strong_Convexity_index.py inpainting 1 120
  ```

### 📊 Figure 2: Global Convergence of PnP

- Demonstrates PnP global convergence with two denoisers:  
  - Symmetric denoiser: `DSG-NLM`
  - Nonsymmetric denoiser: `NLM`
- Code: [`Figure_2_Global_Convergence_inpainting.py`](Figure_2_Global_Convergence_inpainting.py)

#### 💡 Example Notebook

- See [`Example_of_denoiser.ipynb`](Example_of_denoiser.ipynb) for a demo of the `DSG-NLM` denoiser.

---

## ⚙️ Setup Instructions

### 1️⃣ Activate Conda Environment

```bash
# Create a new conda environment with Python 3.8.18
conda create --name myenv python=3.8.18

# Activate the environment
conda activate myenv
```

### 2️⃣ Install Required Packages

Ensure you are in the root directory (where `dependencies.sh` is located):

```bash
sh dependencies.sh
```

Your environment should now be ready with Python 3.8.18 and all required packages installed.

---

## 📚 Citation

```bibtex
@article{sinha2024strongconvexitypnp,
  author  = {Sinha, Arghya and Chaudhury, Kunal N.},
  title   = {On the Strong Convexity of PnP Regularization Using Linear Denoisers},
  journal = {IEEE Signal Processing Letters},
  year    = {2024},
  volume  = {31},
  pages   = {2790-2794},
  doi     = {10.1109/LSP.2024.3475913}
}
```
