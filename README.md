# HousingSalesPrediction
Predicting house sales in King County using a dataset and predictive models.

# Introduction:
We examined King County’s real estate during the period of May 2014 and May 2015. This dataset, encapsulating house sale prices within the region, has been curated to evaluate regression models. The objective is to gain insights into the features influencing housing prices. 

# Summary Statistics: 
After we uploaded the data set, we split our response variable, price, from the input features, and separated our data into training and testing sets so we could start the data analysis process. We started by analyzing the format of each feature, so we could determine what we wanted to use, and the nature of each feature. We split our features into numerical and categorical features, and started our analysis. For the numerical features, we obtained the summary statistics (shown below) and observed that there was a trend to our data: our data appeared to be heavily right-skewed. Upon delving into the numerical features, particularly square-foot living and square-foot lot, noticeable disparities emerged in their summary statistics. Square-foot living exhibited a mean of 2079 with a standard deviation of 918, indicative of considerable variability in the dataset. Conversely, square-foot lot presented a pronounced asymmetry, with a mean of 15,100 significantly surpassing the median of 7620. The substantial standard deviation of 4140 underscored the dispersion of values. The vast difference between the minimum and maximum values accentuated the skewed nature of the distribution, highlighting the broad range of values within these particular features. 

We then shifted our focus to categorical features, and explored the distribution of values through histograms (pictured below) so we could gain a visual understanding of the patterns and frequencies within these qualitative attributes. The categorical features also mirrored the right-skewness of our data. For the condition feature, we noticed that there were lots of gaps. The most common condition rating was "Fair" (3), followed by "Good"(4) and "Excellent" (5). There were very few houses with a condition rating of "Poor". For the histogram of bedrooms, the most common number of bedrooms is 3, and there were very few houses with 1 or more than 5 bedrooms. The floors feature also showed that there were lots of gaps, with the most common number being 1 to 2 bedrooms. For bathrooms, like the other features, the most common number was around 2 to 3 bathrooms. For square-foot living, the most common square footage is between 1,500 and 2,500 square feet. The square foot lot was mostly spread with a very small square footage. 

	Furthermore, we created scatterplots to analyze our data as well (pictured below). The scatterplot depicting square foot living versus price displayed a positive linear relationship, indicating that as the property's square footage increases, its price tends to rise. The correlation coefficient of 0.702 signifies a strong association between these variables, highlighting a clear trend. On the contrary, the scatterplot for square foot lot versus price illustrated a weaker positive relationship compared to the former. While an increase in lot size generally corresponds to higher prices, the correlation coefficient is significantly lower at 0.08, indicating a weaker connection. Despite a few outliers, both scatter plots showcase the overall positive relationship between square footage and price, with square foot living exhibiting a stronger correlation than square foot lot.

# Models:
	In addition to analyzing summary statistics, we developed four regression models to predict housing prices: Linear Regression, Lasso Regression, Cubic Regression, and Gradient Boosting Regression. Our Linear Regression model achieved an R2 score of approximately 0.702 on the training set, showcasing its ability to capture a substantial portion of the variation in housing prices based on input features. The slightly lower R2score of approximately 0.681 on the testing set suggests good generalization to new data. The close alignment between the training and testing set scores indicates that the model maintains robust predictive capabilities beyond the training data. While further optimization may enhance performance, these results affirm the model's proficiency in predicting housing prices across diverse datasets.
The Lasso regression model applied to the house sales dataset demonstrates interesting behavior across various alpha values (shown below). For the initial five alpha values ranging from 0.001 to 10, the coefficients seemed to be relatively stable. However, as alpha increases to 100 and 1000, there was a more pronounced decrease in coefficients. This reduction is attributed to the L1 regularization (Lasso), which pushed some coefficients closer to zero. The R2 values remained consistently high until alpha reached 100, where a slight drop was observed. This decline in R2 occurred because higher alpha values limited the model's ability to precisely fit the data. Despite this, even with relatively high alpha values, the R2 score remained at approximately 0.78, indicating a reasonably good fit of the model to the data. We deduced that smaller alpha values resulted in less regularization, while larger values increased regularization, influencing the shrinkage of coefficients and the overall model performance.


The results for the cubic regression model applied to the house sales data revealed somewhat promising insights. The R2value for the training set, stood at approximately 0.544, and the R2 value for the testing set, approximately 0.531, signified that the model explained a substantial portion of the variance in the target variable. The proximity of these R2 scores for the training and testing sets suggest that the model was not overfitting to the training set, indicating its ability to generalize well to unseen data. This alignment in performance across datasets implies a robust predictive capability. In summary, the cubic regression model appeared to be a somewhat reasonably good fit for the house sales data, demonstrating consistency in its explanatory power and generalization to new observations.
Finally, we chose a new model, Gradient Boosting Regression, and fit it to our data.  Gradient Boosting Regression is a machine learning algorithm that sequentially builds decision trees to minimize prediction errors. It starts with a simple initial model and iteratively adds new trees, each focusing on the errors made by the previous tree. It combines multiple decision trees to predict a numerical value, by correcting errors made by previous trees and using gradient descent for optimization. The model is robust, handles non-linearity, and provides insights into feature importance.
	Upon further analysis of Gradient Boosting Descent, our group found that the  R2 score for the training set was approximately 0.902, which was very close to 1. This high score suggests that the model does a great job of explaining the variance in housing prices. The  R2 score for the testing set was approximately 0.855. Since both scores were relatively high, the model was able to generalize well to new data and showed that the model was not over or underfitting the data.

# Best Model Analysis:
Following the analysis of all models, we concluded that Gradient Boosting Regressor was the best model for 3 reasons. Firstly, this model had strong predictive power. The high R2 score demonstrated that the Gradient Boosting model has the power to predict complex, non-linear relationships between the predictive variables and the house prices. Secondly, as discussed above, similar scores for the  R2 metric could be seen for the training and testing data, suggesting that the model generalizes well to new, unseen data and does not over or underfit the data. Lastly, this model is known for its robustness to outliers in the data, which our data had a lot of. Outliers typically have less impact on the model's predictions compared to simpler models like linear regression. For the following reasons, the Gradient Boosting Model stood out as the best predictive model to gain insights into the factors influencing housing prices.

# Conclusion:
In conclusion, our analysis of King County's real estate data from May 2014 to May 2015 has showcased key insights into housing prices. The positive linear correlation between square footage and price is notable. Our exploration of regression models, including Linear Regression, Lasso Regression, Cubic Regression, and Gradient Boosting, highlighted distinct predictive patterns. The standout performer was the Gradient-Boosting model, showing robust explanatory power with high R2 scores for both the training and testing sets. These findings contribute valuable knowledge to the understanding of housing dynamics in King County, laying the groundwork for further research and model refinement in the real estate domain.


