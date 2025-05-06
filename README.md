# Improvement on Spain Electricity Price Prediction Using Multivariate Linear Regression (Do Not View In Dark Mode)

# Abstract
One of the main issues with electricity markets is predicting renewable energy loads for generators and prices for customers. Due to the advent of machine learning, energy forecasting techniques have improved significantly, but is also in dire need of wide-spread implementation. Forecasted load and price data, as well as actual price data from a four year period between 2015 and 2018 was gathered by fellow Kaggle user Nicholas Jhana, which was collected from ENTSOE (European Network of Transmission System Operators for Electricity) and the Spanish TSO Red Electric España. The goal of this project is to improve upon the day-ahead electicity pricing using multiple linear regression.

Results shown below, using squared mean error as the method of comparison.

* **New Predicition Error: 9.51**
* **TSO Prediction Error: 13.44**

The results show that a simple multiple linear regression using the predictions from the Spanish TSO, and forecasted load data greatly improved the predictions of electricity prices in Spain.


# Problem Statement
Does the current TSO in Spain accurately predict electricity prices? Can we improve the predictions with the provided data? I will use multiple linear regression to improve upon the existing electricity predictions.

# Data Overview
The TSO predictions consistently underestimate the electricity prices, with 89% of the predictions being an underestimate, and a median underestimate of €7.41/MWh. What's interesting, is the profile shape of the actual and predicted prices are very similar, the only difference being that the TSO estimates were shifted down to the left.

![image 1](/png/image5.png)
![image 1](/png/image6.png)

# Method
A multiple linear regression with an 80%-20% data split for training and testing the algorithm. Below are the dependent variables that will be used to predict electricity price.

* **Forecast Solar Day Ahead**
* **Forecast Wind Onshore Day Ahead**
* **Total Load Forecast**
* **Hour of Day**
* **Price Day Ahead (TSO Prediction)**

The TSO predicted price will be used because it may be advantageuous to use their predictions, since the profile of their predicted prices resembles the actual price.

# Results
A multiple linear regression model was implemented, and a score of 54.6% was achieved. This seems quite low, but when compared to the TSO predictions, was a significant improvement over their predictions. The median difference between this method's predictions and the actual price hovered around zero, whereas the median TSO predctions were around €7 below the actual prices.

The multiple linear regression resulted in a lower mean absolute error of, with the results shown below.
* **New Predicition Error: 9.51**
* **TSO Prediction Error: 13.44**

The predicted electricity price profiles from this report and the TSO are compared along with the actual prices. This paper's predictions are more in line with the actual prices, and the price differences show that add to the fact that a simple multiple linear regression improved the predictions of the Spanish TSO.

![image 1](/png/image7.png)
![image 1](/png/image8.png)

# Conclusion
It was shown that a simple multiple linear regression could improve the predicted electricity prices from the Spanish TSO. A combination of using the TSO's predicted prices, as well as the predicted solar, wind, and total load forecasts, and the hour of the day, improved the TSO's predicted electricity prices. Further investigation using more data and a more complicated machine learning algorithm could produce better results.
