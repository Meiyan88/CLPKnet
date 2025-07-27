# Project

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

This  is the pytorch implementation of ‘Contrastive Learning and Prior Knowledge-induced Feature Extraction Network for
Prediction of High-risk Recurrence Areas in Gliomas’

## Key Features

- Medical image data processing 
- Neural network training framework
- ROI  drawing tools
- Data augmentation generation (artificial data generation)
- Grad-CAM visualization

## Directory Structure
├── Dataprocess/ # Data processing scripts
├── Dataset/ # Dataset-related files
├── Iteration/ # Iterative training components
├── Network/ # Network architecture
├── FSL_HCP1065_FA_1mm.nii.gz # The DTI template  
├── train.py # Main training script
├── test1.py # Test script
└── requirements.txt # Python dependencies


## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt


python train.py 

