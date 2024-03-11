# BankChurn 

BankChurn is a project designed to analyze and predict customer attrition (churning) from the [Credit Card Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) utilizing various Machine Learning Algorithms and Data Sampling Techniques. 
Below is a summary of the findings from the notebook.

## Classifiers Used 
This project utilized various Machine Learning Algorithms to classify if a customer would churn or not: 
- KNN (k was chosen to be the sqrt of all samples was chosen, as shown to be best practice [here](https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb?gi=a531f45dfde4#:~:text=The%20optimal%20K%20value%20usually,be%20aware%20of%20the%20outliers)). 
- SGD (i.e. The SGD Classifier in Sklearn)
- Decision Trees
- Random Forest
- SVM
- Soft Voting Ensembles (See here). 
These were chosen primarily for being distinct from one another (except Random Forests and Decision Trees), and primarily to gauge the performance difference between different classical ML algorithms versus an Ensemble Method. 

## The Issue 

The Credit Card Dataset is largely unbalanced, with only 16% of customers being attrited vs 84% of non-attrited customers. 
This presents an issue in attempting to create a classifier for customers, as while classifiers may retain a high accuracy rating(~96% at most without sampling), they perform less adequately in terms of metrics such as recall (~80% at most without sampling). 
In other words, while classifiers may be excellent at predicting whether a customer will churn or not, the **cases in which a customer does are far less accurate than if they do not**. 

In this regard, recall is the primary metric that will be maximized, as doing so would minimize the chance of false negatives in the prediction of customer churn when paired with accuracy. 
However, other metrics such as precision, F1 score, and PR curves were also considered. (More information is detailed in the notebook). 

![A bar graph showing 84 percent of customers being non-attrited for a total of 8500 and 16 percent of customers being attrited for a total of 1627](https://github.com/pw45000/BankChurn/blob/main/images/customer.png?raw=true)

## The Solution
There are many solutions to the issue, but the solution that was employed was the use of Data Sampling, specifically- [Random Over-Sampling](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=FREE%20Mini%2DCourse-,Random%20Oversampling%20Imbalanced%20Datasets,-Random%20oversampling%20involves), [Random Under-Sampling](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=look%20at%20undersampling.-,Random%20Undersampling%20Imbalanced%20Datasets,-Random%20undersampling%20involves), and [SMOTE](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/). 
The sampling techniques were chosen similarly to the classifiers, namely to compare naive sampling methods (i.e. over and under-sampling) to a more complex sampler (SMOTE).

Data Sampling largely mitigates the issues of an unbalanced dataset by simply balancing it. Depending on the sampler, the majority class is undersampled (i.e. culled), or the minority class receives artificial copies (oversampling, done in a slightly more sophisticated way in SMOTE).

## Methods for Training and Evaluation
Various multiprocessing steps such as removing certain multicollinear columns, standardizing the dataset, and more were undertaken. 

Afterward, various copies of the dataset were made and had the various sampling techniques applied to them. The modified datasets were then split into testing and training sets. 

Separate classifiers were then trained on each of the different sampling methods. In total, 23 different classifiers were trained. 
(Note that the last sampler was a Random Forest outside of this procedure, made to test feature importance).


## Results 

### Samplers 
- Oversampling typically retained high accuracy and overall f1 score, but lower recall.
- SMOTE was not particularly great at anything, but paired with the Random Forest classifier was tied with the highest f1 score. 
- Undersampling typically retained a high recall, but lower rates of accuracy and f1 score. 
- No sampler had the highest accuracy as stated earlier, but the lowest recall. 

### Classifiers 
- Overall, KNN and SGD were quite quick but inadequate, with a ~70-80% recall but low precision scores and produced the worst metrics on average.
- Decision Trees were slightly better in Recall and F1 score, albeit slower. 
- SVM showed promise with a high recall score (90% with Oversampling and Undersampling) but was the slowest to fit and had a nearly random precision (~60% at lowest). 
- Random Forests were the best overall in terms of f1 score(~80%) and also retained a 94% or higher accuracy. This classifier, when paired with multiple samplers, had nearly all the highest values in terms of metrics. 

In addition, two soft-voting ensembles were made taking three models which were either the best overall or best in recall: 
- Ensembling the best of Recall: the whole was lesser than the sum of its parts in metrics(SVM Over and Undersampled, Random Forest Undersampled)
- Ensembling the best of overall stats closely resembles the performance as Random Forest Oversampled (Random Forest Under, Over, SMOTE sampled).

### What is the best combination? 
- Random Forest (Under Sampling) had a 95% recall and 94% accuracy, so by the metrics defined earlier, it’s the “best” classifier.
- While it has impressive primary metrics, it suffers from poor precision, but false positives are the lesser evil compared to a false negative.
![The metrics of the Random Forest Classifier. It has a 95% recall and 94% accuracy, as well as 77% precision and an 85% f1 score.](https://github.com/pw45000/BankChurn/blob/main/images/metrics.png?raw=true)
![The confusion matrix of the Random Forest Classifier. It has 1604 true positives, 310 true negatives, 95 false positives, and 17 false negatives.](https://github.com/pw45000/BankChurn/blob/main/images/confusion_matrix.png?raw=true)

## What Contributed the Most to Churning? 
Utilizing the best classifier (i.e. Random Forest Under Sampling), feature importance can be obtained via a sklearn method on said classifier. 

Of the features explored, Total Transaction Count and Total Revolving Balance are the most telling in terms of risk factors. 

![The graph of each feature's importance. Here, outside of the previously mentioned factors, the revolving balance and change from quarter 4 to 1 are the most important.](https://github.com/pw45000/BankChurn/blob/main/images/feature_importance.png?raw=true)


## Contributors 
None of this would have been possible without [tortega-24](https://github.com/tortega-24) who helped provide the Explanatory Data Analysis for the notebook, insights into preprocessing the dataset, and more. 

