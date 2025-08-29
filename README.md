# <ins>TitanicProject</ins>

# <ins>**1. Project Overview**</ins>
This project demonstrates an end-to-end MLOps pipeline for a classification task using the Titanic dataset. It covers distributed data preprocessing, model training with Spark MLlib, experiment tracking with MLflow, and a basic model serving API. The pipeline is designed to be scalable, automated, and reproducible, addressing key challenges in real-world machine learning operations.

# <ins>**2. Prerequisites**</ins>
To run this project, you need to have the following installed on your system:

  1. **WSL2:**
  The development environment is WSL2 because it provides better support for Spark.
    
  2.   **Docker Desktop (with WSL2 backend):**
This is essential for running the project in a containerized environment, which ensures consistency and reproducibility.

  3. **Git:**
For cloning the repository.

  4. **Conda:** 
To manage the project's Python environment.

  5. **DVC (Data Version Control):**
For managing and versioning the dataset.

# <ins>**3. Setup Instructions**</ins>
  1. WSL Setup:
Open wsl from windows.
Install conda, create a new environment and activate it:


  3. 
