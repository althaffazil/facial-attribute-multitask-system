# Facial Attribute Multitask System

A production-ready deep learning system for simultaneous facial attribute prediction using a spatially decoupled multi-task CNN architecture with integrated model explainability and containerized deployment.

This project demonstrates end-to-end ML system design — from architecture refinement and task disentanglement to explainable inference and reproducible Docker deployment.


## Key Capabilities

* Multi-task CNN (Gender + Smile)
* Spatially decoupled feature branching
* Independent classifier heads
* Task-specific Grad-CAM explainability
* Device-agnostic execution (CPU/GPU)
* Modular codebase
* Fully containerized deployment
* Interactive Gradio interface


## Architecture Design

The model leverages a pre-trained MobileNetV2 backbone with:

* Shared low-level feature extractor
* Task-specific convolutional adapters
* Independent classification heads
* Decoupled gradient flow for explainability

This design reduces cross-task feature interference while preserving computational efficiency.

### Engineering Focus

* Reduced task entanglement
* Controlled gradient flow
* Clean modular structure
* Production-consistent runtime



## Model Performance

| Task   | Accuracy | F1 Score |
| ------ | -------- | -------- |
| Gender | ~98.5%   | ~0.98    |
| Smile  | ~92–93%  | ~0.92    |

Evaluation performed on validation split.



## Application Preview



<img width="1408" height="821" alt="demo" src="https://github.com/user-attachments/assets/7e730e33-2e4a-4063-9143-52a66437eefb" />


## Run with Docker (Recommended)

The system is available as a public Docker image.

### Pull the image

```bash
docker pull cybervamp/facial-attribute-multitask-system:latest
```

### Run locally

```bash
docker run --rm -p 7860:7860 cybervamp/facial-attribute-multitask-system:latest
```

Then open:

```
http://localhost:7860
```

The application will start with CPU inference by default.



## Tech Stack

* Python
* PyTorch
* Torchvision
* Gradio
* scikit-learn
* OpenCV
* Docker



## Project Structure

```
facial-attribute-multitask-system/
│
├── configs/
│   └── config.py
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── multitask_model.py
│   └── best_model.pth
│
├── utils/
│   ├── gradcam.py
│   ├── inference.py
│   ├── metrics.py
│   └── trainer.py
│
├── screenshots/
│   └── demo.png
│
├── app.py
├── train.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

The project is structured for scalability, maintainability, and reproducible deployment.


## Engineering Highlights

* Designed and trained a spatially decoupled multi-task CNN
* Integrated task-level Grad-CAM explainability
* Reduced cross-task feature interference
* Optimized Docker image size (CPU-only torch)
* Ensured NumPy / PyTorch ABI compatibility
* Built a fully reproducible deployment pipeline

