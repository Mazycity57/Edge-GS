# Edge-GS: Edge-Aware Gaussian Splatting for Enhanced 3D Scene Rendering

**Edge-GS** is a novel framework for large-scale 3D scene reconstruction based on 3D Gaussian Splatting. It enhances [Scaffold-GS](https://github.com/kaixin96/Scaffold-GS) by introducing edge-aware improvements such as an Anchor Re-Growing Module and SAM-based segmentation loss. These components significantly improve the fidelity of rendered edges, especially in complex indoor and outdoor environments.

---

## ✨ Key Features

- **Edge-Aware Anchor Re-Growing**  
  Dynamically inserts Gaussians near image gradients to increase detail around object contours and high-frequency regions.

- **SAM-Based Segmentation Loss**  
  Leverages the Segment Anything Model (SAM) to compare rendered and ground truth masks, encouraging alignment of structural boundaries.

- **Content-Aware Scene Partitioning**  
  Scene is automatically divided based on point density and view coverage, enabling efficient and scalable multi-GPU training.

- **Built on Scaffold-GS**  
  Inherits efficient large-scale Gaussian rendering design, and is compatible with the original Scaffold-GS pipeline.

---

## 🛠 Installation

```bash
git clone https://github.com/Mazycity57/Edge-GS.git
cd Edge-GS
conda create -n edgegs python=3.10
conda activate edgegs
pip install -r requirements.txt
