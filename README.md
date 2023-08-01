# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: AI Squat Training Assistant

### **Try Out AI Squat Streamlit Application by clicking the link below.**
# [AI Squat Training Assistant](https://ai-squat-assistant.onrender.com/)
Project management and planning documentation is done via Github Projects here: https://github.com/khammingfatt/project-5-ai-enhance-squat-form

<br>

| **The 'Rest' State** | **The 'Down' State**  |
| ------------------------ | -----------------------  |
| ![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Readme%20table%201.jpg?raw=true)| ![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Readme%20table%202.jpg?raw=true) |

<br>

## Content Directory:
- [Background](#Background)
- [Data Import](#Data-Import)
- [Manual Modeling](#Manual-Modeling)
- [Modeling and Hyperparameters Tuning](#Modeling-and-Hyperparameters-Tuning)
- [Model Evaluation](#Model-Evaluation)

<br>


## Background
![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/straits_times.jpg?raw=true)

The **Sit to Stand Test** serves as an assessment tool for evaluating leg strength and endurance among older adults. It is an integral component of the Fullerton Functional Fitness Test Battery. The development of this test aimed to address the floor effect encountered in the five or ten repetition sit to stand test when applied to older adults.

![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/sit-stand-test.jpg?raw=true)

The muscle groups engaged during both the sit to stand test and squats exhibit **considerable similarity, encompassing the legs, thighs, and back muscles**. 

![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/muscle-group.jpg?raw=true)

**Therefore, engaging in squat training can significantly enhance performance in the sit-and-stand test, thereby contributing to individuals' overall longevity and well-being.**

Reference Website
- [Simple sit-and-rise test predicts how long you'll live](https://www.straitstimes.com/singapore/health/simple-sit-and-rise-test-predicts-how-long-youll-live)
- [Why Squats Are The King of All Exercises](https://medium.com/@leanwaistwarrior/why-squats-are-the-king-of-all-exercises-4621119900b0)

<br>



## Problem Statement
### The project aims to empower individuals to improve their health and fitness by offering an AI option to 

### (i) perform correct squats independently without an exercise coach;

### (ii) assess their squat performance;

### (iii) enhance their squat performance by using AI to provide prescriptive feedback to the squat form.

![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Problems.jpg?raw=true)

<br>

---

## Datasets:
Public health workers in Chicago setup mosquito traps scattered across the city. The captured mosquitos are tested for the presence of West Nile virus.

* [`coords.csv`](../assets/train.csv): The "coords.csv" dataset comprises coordinates of dataset collected 

<br>

## Data Dictionary
| Feature | Type | Dataset | Description |
| :--- | :--- | :--- | :---|
| class | str | coords | the label of the coordinates |
| x1 | int | coords | x-coordinate of the landmark 1: nose |
| y1 | int | coords | y-coordinate of the landmark 1: nose |
| z1 | int | coords | z-coordinate of the landmark 1: nose |
| v1 | int | coords | visibility of the landmark in computer vision |
| ... | ... | ... | ... |
| x33 | int | coords | x-coordinate of the landmark 33: nose |
| y33 | int | coords | y-coordinate of the landmark 33: nose |
| z33 | int | coords | z-coordinate of the landmark 33: nose |
| v33 | int | coords | visibility of landmark 33 in computer vision |

<br>

![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/landmarks.jpg?raw=true)
<br>
With reference to the diagram above, we have 33 landmarks and 133 columns of data in total. 


<br>
<br>

## Data Collection
The image below is a pictorial illustration of the data collection process.
![Data Collection](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Data%20Collection.jpg?raw=true)

The data collection process involves accessing the webcam using **OpenCV (Open Source Computer Vision Library)** and utilizing the **MediaPipe library** for landmark detection. The objective is to collect body pose data in real-time and save it in a structured format.

The code starts by setting up the video capture device (webcam), and then the Holistic model from MediaPipe is initialized with specific confidence thresholds for detection and tracking. The while loop allows continuous video streaming, capturing each frame from the webcam. The captured frame is recolored to RGB for further processing.

The Holistic model is then applied to the frame to detect various **landmarks, including face, hands, and body pose**. The detected landmarks are drawn on the frame using different colored lines and points, enhancing the visualization of the captured data.

The essential part of the data collection process involves exporting the landmark coordinates to a CSV file. The coordinates are extracted from the detected pose landmarks, including the x, y, z positions, and visibility values for each landmark. These values are flattened and saved as a row in the **'coords.csv'** file. Additionally, the data is associated with a class name, which could be a label for a specific action or pose.


## Modeling

![Modeling](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Model.jpg?raw=true)

Following the data preprocessing stage, the preprocessed dataset was fed into a pipeline of five classification models: **logistic regression, random forest classifier, XGBoost, Adaboost, and Voting Classifier**. These models were carefully chosen based on their respective strengths and suitability for the classification task at hand. The use of multiple models allowed us to leverage their individual capabilities and ensemble them to make more robust predictions.

By employing this systematic approach, we aimed to enhance the quality and reliability of our classification results, enabling us to make informed decisions based on the predictions generated by the ensemble of models.


<br>
<br>

## Summary of Model Perforamance

|  | Train Accuracy | Test Accuracy | Train F1 Score | Test F1 Score |
|---|---|---|---|---|
| **Null Model <br>(Baseline Model)** | 0.5 | 0.5 | 0.5 | 0.5 |
| **Random Forest Classifier** | **0.9983** | **0.9960** | **1.0** | **0.9957** |
| **Logistic Regression** | 0.9983 | 0.9960 | 1.0 | 0.9958 |
| **XGBoost Classifier** | 0.9983 | 0.9960 | 1.0 | 0.9958 |
| **AdaBoost Classifier** | 0.9965 | 0.9960 | 1.0 | 0.9957 |

<br>


## Key Recommendations

Leveraging the power of machine learning, we have developed an advanced predictive model to effectively correct the squat posture. We invite you to delve into the comprehensive details of our proposal outlined below, which showcase the technical prowess and formal methodologies employed in our solution.

| **Prescription Messages** | **Problems to be Solved**  |
| ------------------------ | -----------------------  |
|![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Squat%20Training.png?raw=true)|![Alt text](https://github.com/khammingfatt/project-5-ai-enhance-squat-form/blob/main/images/Prescriptions.jpg?raw=true) |

<br>

## (1) Learn Correct Squat **Independently**

## (2) **Reduce** the chance of injury by performing squat correctly

## (3) **Assess Your Squat Performance** everywhere 


---
## Reference
(1) Straits Times: Simple Sit-and-Rise Test Predicts How Long You'll Live <br>
https://www.straitstimes.com/singapore/health/simple-sit-and-rise-test-predicts-how-long-youll-live

(2) Why Squats Are The King of All Exercises <br> 
https://medium.com/@leanwaistwarrior/why-squats-are-the-king-of-all-exercises-4621119900b0

(3) Health Benefits of Squats <br> 
https://www.webmd.com/fitness-exercise/health-benefits-of-squats#:~:text=Squats%20burn%20calories%20and%20might,ligaments%20around%20the%20leg%20muscles.

(4) Mediapipe Documentation
<br> https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
