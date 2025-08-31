# <ins>TitanicProject</ins>

# <ins>**1. Project Overview**</ins>
The primary goal of this project is to design, implement, and document a comprehensive MLOps pipeline. To achieve this various tools and methodologies have been implemented.

# <ins>**2. Prerequisites**</ins>
Following prerequisites are required to run the project.

  1. **WSL2:**
    
  2.   **Docker Desktop (with WSL2 backend):**

  3. **Git:**

  4. **Conda:** 

  5. **DVC (Data Version Control):**

# <ins>**3. Setup Instructions**</ins>
### 1. WSL Setup
First, open your Windows Subsystem for Linux (WSL) terminal.

-   **Install and Activate a Conda Environment:**

    ```bash
    conda create --name lab_titanic_env python=3.10
    conda activate lab_titanic_env
    ```

### 2. Clone the Repository

Navigate to the directory and clone the repository with the command. 

```bash
git clone "https://github.com/ch24m519-rgb/ai-lab-titanic-dataset.git"
```
###  3. Install packages:
  Navigate inside AI-Lab folder
  ```bash
  cd TitanicProject
  pip install -r requirements.txt
  ```

###  4. Pull data from dvc 
   ```bash
   dvc pull
   ```
  
  It will pull all the pipelines, models are its artifacts.

###  5. Putting data
   As the data is required to analyse the result and run the project. Kindly place train.csv and test.csv inside TitanicProject/data/raw folder 
  
### 6. In case model is available run only eval.py with test.py script.
    
   ```bash
   python src/eval.py
   ```

  ### 7. For training the model following commands to be run
  
1. Run
   ```bash
     mlflow ui
   ```
Take seperate terminal and activate the conda environmenet with 
  '''bash
     wsl
     conda activate lab_titanic_env
    ```
  '''bash
 2.   python src/preprocessing_spark.py
 3.   python src/train_spark.py 
   '''
   
  Go to Web Browser with the address:
  http://localhost:5000/
  Select "Titanic-Project" from experiments to see running experiments.
  
  ### 7. Run Evaluation file to accept the input:
    
   ```bash
   python src/eval.py
   ```
    
    In another terminal (in wsl, in same environment)
    
   ```bash
   python src/test.py 
   ```
    
    [
      Prediction as 1 for Survived and 0 for Not-Survived.
    ]

  ### 8. Deployment at Docker & Containerization
  Create a docker image (titanic_docker_img):
  
  ```bash
  docker build -t titanic_docker_img .
  ```
  
  Run a docker container (titanic):
  
   ```bash
    docker run -p 5050:5050 -d --name titanic titanic_docker_img
   ```
  
  To check the logs:
  
   ```bash
   docker logs titanic
   ```
  
  To stop container:
  
   ```bash
   docker stop titanic
   ```

