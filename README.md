# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

How to install and run the code:
1)Clone the code to your local
2)Use the env.yaml and create the environment with required packages for the subsequent codes to run
3)install the Package_Housing-0.1-py3-none-any.whl file in the dist folder 
4)This will unpack all the required python codes required. 
5)In the terminal run python
6)In Python you can now from Package_Housing import ingest_data,train_data
7)Each of the individual modules could also be run with the parameters path and datasets name
8)To run ingest_data -i /location of the processed file to be saved
9)To run train.py -i /location of the processed file /location of model artifacts
10)To run score.py -i /location of model artifacts  /location of the processed file

 
