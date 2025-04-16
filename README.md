# VectorSync: A Deep Learning Approach for IoT Device Localisation
Explore the cutting-edge world of Machine Learning (ML) algorithms in wireless communications with our research project. Our study extensively analyzes ML algorithms, focusing on their time consumption and positional accuracy compared to traditional methods. The spotlight is on predicting the Reconfigurable Intelligent Surface (RIS) phase using channel coefficients as input.

Our standout model, the VectorSync Beamformer Model, trained on NSB, outshines conventional algorithms in antenna weight prediction, offering superior performance and reduced computation time on advanced devices. While demonstrating rapid calculations for User Equipment (UE) positioning, there's a noted trade-off with positional accuracy compared to state-of-the-art procedures. This intriguing balance sets the stage for future enhancements, aiming to refine the VectorSync Beamformer Model and strike the optimal equilibrium between speed and accuracy in detecting UE positions.

Join us on this journey of innovation and exploration!

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Project Directory](#project-directory)
- [Results](#results)
- [References](#references)
    
## Installation

#### Prerequisites

First make sure you have [Conda](https://docs.conda.io/en/latest/) installed on your system. Neither you may setup your environment using pip. 

##### Clone The Repo
```bash
git clone https://github.com/omm-prakash/VectorSync.git
```
##### Move Into The Repository
```bash
cd ./VectorSync
```
##### Create Conda Environment
Use the provided `environment.yml` file to create a Conda environment named "grocery".
```bash
conda env create -f environment.yml
```
##### Activate Conda Environment
```bash
conda activate VectorSync
```

## Features

- **RIS Phase Prediction:** .
- **Antenna Weight Prediction:** Used `VectorSync` to determine antenna weight inplace of `NSB` and `Capon` methods.

## Project Directory
```html
<!-- root directory -->
.
├── Deep-Learning
│   ├── autoencoder_weights <!-- trained model weights -->
│   │   ├── azimuthal
│   │   │   ├── lstmattention/
│   │   │   ├── lstmmodel/
│   │   │   ├── model1/
│   │   │   ├── model2/
│   │   │   └── model3/
│   │   ├── azimuthal_position
│   │   │   ├── lstmattention/
│   │   │   ├── lstmmodel/
│   │   │   ├── model1/
│   │   │   └── model2/
│   │   └── position
│   │       ├── lstmattention/
│   │       ├── lstmmodel/
│   │       ├── model1/
│   │       └── model2/
│   ├── dataloader_gpu.py
│   ├── loss.py
│   ├── models.py <!-- model architectures -->
│   ├── __pycache__/
│   ├── README.md
│   ├── train_checkpoint.py
│   ├── train_gpu.py
│   └── train_lstm.py
├── environment.yml <!-- environment setup -->
├── README.md
└── Simulation
    ├── beamforming.ipynb <!-- analysis of beforming -->
    ├── capon.py
    ├── data/ <!-- binary/csv files -->
    ├── DoAEstimation.py 
    ├── doa.py
    ├── models.py <!-- model architectures -->
    ├── model_states/ 
    │   ├── model_weights_best_198.pt
    │   └── model_weights_best_97.pt
    ├── music.ipynb
    ├── nsb.ipynb <!-- data generation for model training -->
    ├── simulation.ipynb <!-- testing on scinario -->
    ├── time_and_coorelation_analysis.ipynb
    ├── utils.py
    └── weight.py
```

## Results
| Model              | Train Loss | Val Loss | Test Loss |
| ------------------ | ---------- | -------- | --------- |
| **VectorSync**     | **3.55e-03** | **3.42e-03** | **3.49e-03** |
| Base Model         | 6.92e-03    | 6.97e-03  | 6.93e-03  |
| Autoencoder        | 8.83e-03    | 8.89e-03  | 8.91e-03  |
| DNN                | 2.44e-02    | 2.37e-02  | 2.55e-02  |

*Table 1: With Angle as Input*

| Model              | Train Loss | Val Loss | Test Loss |
| ------------------ | ---------- | -------- | --------- |
| **VectorSync**     | **2.92e-03** | **2.85e-03** | **2.85e-03** |
| Base Model         | 3.02e-03    | 3.23e-03  | 3.19e-03  |
| Autoencoder        | 7.49e-03    | 7.43e-03  | 7.15e-03  |
| DNN                | 2.45e-02    | 2.44e-02  | 2.41e-02  |

*Table 2: With Position as Input*

| Model              | Train Loss | Val Loss | Test Loss |
| ------------------ | ---------- | -------- | --------- |
| **VectorSync**     | **2.20e-03** | **2.17e-03** | **2.35e-03** |
| Base Model         | 2.66e-03    | 2.75e-03  | 2.62e-03  |
| Autoencoder        | 7.86e-03    | 8.44e-03  | 7.95e-03  |
| DNN                | 2.23e-02    | 2.22e-02  | 2.26e-02  |

*Table 3: With Angle and Position as Input*


## References


For more details or any clarifications please feel free to contact me @ gaganvishalmundada@gmail.com .

Best Regards,<br>
Gagan Mundada <br>
Omm Prakash Sahoo
