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


<br>

---

## Datasets:
Public health workers in Chicago setup mosquito traps scattered across the city. The captured mosquitos are tested for the presence of West Nile virus.

* [`train.csv`](../assets/train.csv): The "train.csv" dataset comprises information regarding the geographical coordinates of mosquito traps, the count of mosquitos captured in each trap, and the presence or absence of the West Nile Virus. The dataset encompasses data collected during the years 2007, 2009, 2011, and 2013.

* [`test.csv`](../assets/test.csv): The "test.csv" dataset comprises information regarding the geographical coordinates of mosquito traps and the count of mosquitos captured in each trap. The dataset encompasses data collected during the years 2008, 2010, 2012, and 2014. We are to use test.csv to evaluate the results of machine learning.

* [`spray.csv`](../assets/spray.csv): The spray.csv consists of GIS data for City of Chicago spray efforts in 2011 and 2013. 

* [`weather.csv`](../assets/weather.csv): The weather.csv consists of weather condition data collected by National Oceanic and Atmospheric Administration (NOAA) from year 2007 to 2014.

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

<br>
With reference to the diagram above, we have 33 landmarks and 133 columns of data in total. 

---

<br>
<br>

## Data Preprocessing
![Data Preprocessing](https://github.com/khammingfatt/project-4-data-backed-solutions-for-combating-wnv-in-chicago/blob/main/images/Data%20Preprocessing.jpg?raw=true)

We followed a rigorous data preprocessing pipeline consisting of three key steps. Firstly, we employed the **One Hot Encoder** technique to convert categorical data into **numerical format**, ensuring compatibility with our models. This transformation allowed us to effectively capture the information contained within the categorical variables.

Next, we applied the **Synthetic Minority Oversampling Technique (SMOTE)** to balance the target class distribution, aiming for a **50:50 ratio**. By oversampling the minority class, we addressed the issue of class imbalance and improved the performance of our models in handling the target variable.

In the final step of data preprocessing, we utilized the **Standard Scaler** method. This process involved transforming all features within the dataset to a **similar scale and distribution**. By doing so, we minimized the potential impact of varying feature magnitudes, allowing our models to better understand the relative importance of different features during the classification process.

## Modeling

![Modeling](https://github.com/khammingfatt/project-4-data-backed-solutions-for-combating-wnv-in-chicago/blob/main/images/Model%20Workflow.jpg?raw=true)

Following the data preprocessing stage, the preprocessed dataset was fed into a pipeline of five classification models: **logistic regression, random forest classifier, XGBoost, Adaboost, and Voting Classifier**. These models were carefully chosen based on their respective strengths and suitability for the classification task at hand. The use of multiple models allowed us to leverage their individual capabilities and ensemble them to make more robust predictions.

By employing this systematic approach, we aimed to enhance the quality and reliability of our classification results, enabling us to make informed decisions based on the predictions generated by the ensemble of models.


<br>
<br>

## Summary of Model Perforamance

|  | TPR<sup>1</sup> | TNR<sup>2</sup> | ROC(Train) | ROC(Test) |
|---|---|---|---|---|
| **Logistic Regression <br>(Baseline Model)** | 0.7802 | 0.7419 | 0.8670 | 0.8269 |
| **Random Forest Classifier** | 0.8352 | 0.7069 | 0.8597 | 0.8552 |
| **XGBoost Classifier** | 0.1319 | 0.9853 | 0.9238 | 0.8733 |
| **AdaBoost Classifier** | **0.7253** | **0.8124** | **0.8259** | **0.8564** |
| **Voting Classifier** | 0.6703 | 0.8443 | 0.8987 | 0.8645 |

<br>
1 - True Positive Rate or Sensitivity.

2 - True Negative Rate or Specificity.  
3 - Public Score on Kaggle.com (AUC)


---

<br>


## Key Recommendations

![Potential Cost Reduction](https://github.com/khammingfatt/project-4-data-backed-solutions-for-combating-wnv-in-chicago/blob/main/images/Potential%20Cost%20Reduction.jpg?raw=true)

Leveraging the power of machine learning, we have developed an advanced predictive model to effectively combat the West Nile Virus. This innovative approach enables us to significantly reduce costs by an impressive 78.5%. We invite you to delve into the comprehensive details of our proposal outlined below, which showcase the technical prowess and formal methodologies employed in our solution.

| **Proposal** | **Proposal Evaluation**  |
| ------------------------ | -----------------------  |
|![Alt text](https://github.com/khammingfatt/project-4-data-backed-solutions-for-combating-wnv-in-chicago/blob/main/images/Proprosal.jpg?raw=true)|![Alt text](https://github.com/khammingfatt/project-4-data-backed-solutions-for-combating-wnv-in-chicago/blob/main/images/Evaluation%20of%20Proposal.jpg?raw=true) |

<br>

## (1) **Time** your effort
- May to October: Economist Approach
- November to April: Minimalist Approach

## (2) **Focus** on culex pipiens populated areas

## (3) **Initiate** localised social media campaign 


---
## Reference
(1) Center for Disease Control and Prevention <br>
https://www.cdc.gov/westnile/statsmaps/historic-data.html

(2) VDCI Mosquito Management <br> 
https://www.vdci.net/

(3) National Library of Medicine
<br> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3945683/

(4) American Society of Tropical Medicine
<br> https://astmhpressroom.wordpress.com/journal/february-2014/

