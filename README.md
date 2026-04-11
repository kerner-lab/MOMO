<p align="center">
  <img src="assets/momo-title.png" alt="MOMO: Mars Orbital Model Foundation Model for Mars Orbital Applications" width="97%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.02719">📄 Paper</a> |
  <a href="#">🤗 HuggingFace (coming soon)</a> |
  <a href="#">📦 Model Checkpoints (coming soon)</a>
</p>

<p align="center">
  Mirali Purohit<sup>1,2</sup><sup>†</sup>, Bimal Gajera<sup>1*</sup>, Irish Mehta<sup>1*</sup>, Bhanu Tokas<sup>1*</sup>,<br/>
  Jacob Adler<sup>1</sup>, Steven Lu<sup>2</sup>, Scott Dickenshied<sup>1</sup>, Serina Diniega<sup>2</sup>,<br/>
  Brian Bue<sup>2</sup>, Umaa Rebbapragada<sup>2</sup>, Hannah Kerner<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Arizona State University &nbsp;&nbsp;
  <sup>2</sup>Jet Propulsion Laboratory, California Institute of Technology<br/>
  <sup>*</sup>Equal Contribution &nbsp;&nbsp; <sup>†</sup>Corresponding Author
</p>

---

## Introduction

We introduce **MOMO**, the first multi-sensor foundation model for Mars remote sensing. MOMO uses model merging to integrate representations learned independently from three key Martian orbital sensors: **HiRISE**, **CTX**, and **THEMIS**; spanning resolutions from 0.25 m/pixel to 100 m/pixel.

Central to our method is a novel **Equal Validation Loss (EVL)** strategy, which aligns checkpoints across sensors based on validation loss similarity before fusion via task arithmetic. This ensures models are merged at compatible convergence stages, leading to improved stability and generalization.

MOMO is trained on ~12 million Mars orbital samples and evaluated on 9 downstream tasks from [Mars-Bench](https://arxiv.org/abs/2510.24010). It outperforms ImageNet pre-trained, Earth observation foundation model, sensor-specific pre-training, and fully-supervised baselines — with particularly consistent gains on segmentation tasks.

<p align="center">
  <img src="assets/momo-teaser.png" alt="MOMO can be applied across a wide range of resolutions and Martian remote sensing tasks." width="90%"><br>
  <em>MOMO can be effectively applied across a wide range of resolutions and a broad spectrum of Martian remote sensing tasks, including large-scale crater or landslide mapping and precise boulder localization.</em>
</p>

---

## Installation

```bash
# Install the package with core dependencies
pip install -e .

# Install with development dependencies (for testing, linting, etc.)
pip install -e ".[dev]"
```

> Requires Python 3.10+ and CUDA 12.x for GPU support.

---

## Usage

> Coming soon. Example commands for pre-training, model merging, and fine-tuning will be added here.

---

## Citation

If you use MOMO in your research, please use the following citation:

```bibtex
@inproceedings{purohit2026momo,
    title={MOMO: Mars Orbital Model Foundation Model for Mars Orbital Applications},
    author={Mirali Purohit and Bimal Gajera and Irish Mehta and Bhanu Tokas and Jacob Adler and Steven Lu and Scott Dickenshied and Serina Diniega and Brian Bue and Umaa Rebbapragada and Hannah Kerner},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2026},
    url={https://arxiv.org/abs/2604.02719}
}
```

### Contact Information

Please reach out to Mirali Purohit [mpurohi3@asu.edu](mpurohi3@asu.edu), if you have any queries or issues regarding MOMO.
