# Advanced-Data-Analysis-Mining-Project
A group project for the Advanced Data Analysis and Machine Learning course 

We choose the project Mining Process A2:

## Level A2 (30p) Future predictions of silica content
Using only lagged variables and past known values of silica ore, calibrate a model that
can estimate the future values of silica content in the ore. You can use dynamic models.
Evaluate the most important variable for prediction.
Note: Since it is a A-level task, you cannot use the outlet iron concentration as an input
variable to the model (unless it is a lagged variable).

We can find the description of the task here: https://moodle.lut.fi/pluginfile.php/2642734/mod_page/content/48/ADAML23%20-%20Mining%20Process%20Quality%20Project%20Work.pdf

## Dataset

The dataset originates from a mining process, from the froth flotation phase. The process is
successful if there is as little outlet silica content as possible. There are 24 variables
(columns) in the data. 

Here is the link to the dataset: https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process

## Week 1 task 

Breakdown of the points:

    0.25p   – an established communication channel and appropriate strategy for code sharing.
    0.25p   – data correctly imported into appropriate matrices completely: observations as rows, variables (predictors) as columns.
    0.5p     – identification of challenges of the data: for example: time series not synchronized, missing values in data, extra variables, variables with unknown physical meanings, etc.
    0.5p     – a visualization and comment on the dataset: variable distribution, number of observations, type of measurements (time series or not time series)
    3p         -  exploratory data analysis with PCA: explain variable correlations and visualize the PCs using biplots, loading plots; (! only on the X matrix - we are not looking at the response variable now)
    0.5p     – identification of pretreatment steps, and a plan on how to do data pretreatment

-We established communication, with a Teams group, and this Github repository to share our code.
-We dowloaded, imported and used the csv file in both of our scripts
-We analyzed the data prior to the use of PCA, in the script polished_up_data_analysing to determine the challenges
-We went through PCA in PCA.py 
-We identified in the reports pretreatments of the steps
