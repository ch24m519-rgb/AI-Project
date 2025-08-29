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
### 1. WSL Setup
First, open your Windows Subsystem for Linux (WSL) terminal.

-   **Install and Activate a Conda Environment:**

    ```bash
    conda create --name titanic_env python=3.10
    conda activate titanic_env
    ```

-   **To remove the environment later:**

    ```bash
    conda env remove --name "titanic_env"
    ```

### 2. Clone the Repository

Navigate to the directory where you want to clone the repository and run the following command. 

```bash
git clone "https://github.com/ch24m513/TitanicProject.git"
```
To remove conda environment:
```bash
conda env remove --name "clone_env"
```
###  3. INstall packages:
  Navigate inside TItanicProject folder
  ```bash
  cd TitanicProject
  pip install -r requirements.txt
  ```

###  4. Pull data from dvc 
(You need to copy the data manually as I have created the dvc locally)
   ```bash
   dvc pull
   ```
  
  It will pull all the pipelines, models are its artifacts.
  
  ### 5. In case I have models readily available with me (downloaded from dvc), I can directly run the eval.py file and can test with test.py script.
    
   ```bash
   python src/eval.py
   ```

  ### 6. As you dont have models, run the following scripts:
  
   '''bash
   python src/titanic_preprocessing.py
   python src/train.py 
   '''
    
  Before running train.py, make sure mlflow in already running in another terminal. It it is not running, run it with the following command:
    
  ```bash
  mlflow ui
  ```
      
  Go to Web Browser with the address:
  http://localhost:5000/
  Select "Titanic-Project" from experiments in top-left corner; All the models will be dispalyed here.
  
  ### 7. Run Evaluation file to accept the input:
    
   ```bash
   python src/eval.py
   ```
    
    In another terminal (in wsl, in same environment. Refer step 1 to change the environment)
    
   ```bash
   python src/test.py 
   ```
    
    [
      You will get the prediction as 1 (Survived) or 0 (Not-Survived).
    ]

  ### 8. Deployment at DOcker & Containerization
  Create a docker image (titanic_docker_img):
  
  ```bash
  docker build -t titanic_docker_img .
  ```
  
  Run a docker container (titanic-cont):
  
   ```bash
    docker run -p 5050:5050 -d --name titanic-cont titanic_docker_img
   ```
  
  To check the logs:
  
   ```bash
   docker logs titanic-count
   ```
  
  To stop container:
  
   ```bash
   docker stop titanic-count
   ```

