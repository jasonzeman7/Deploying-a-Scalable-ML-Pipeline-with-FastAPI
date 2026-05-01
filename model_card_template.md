# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn. It was developed
as part of a machine learning pipeline project. The model predicts whether an individual
earns more or less than $50,000 per year based on census data. The model uses default
hyperparameters with a random state of 42 for reproducibility.

## Intended Use

This model is intended to predict income levels (>50K or <=50K) based on demographic
and employment information from the U.S. Census Bureau dataset. It is intended for
educational purposes and should not be used for making real-world financial or hiring
decisions.

## Training Data

The model was trained on the Census Income dataset from the UCI Machine Learning
Repository. The dataset contains 32,561 rows of census data. The data was split 80/20
for training and testing. Categorical features were encoded using a OneHotEncoder and
the label was binarized using a LabelBinarizer.

The categorical features used are:
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

## Evaluation Data

The model was evaluated on 20% of the original dataset, which was held out during
training. The same encoder and label binarizer fitted on the training data were used
to process the evaluation data.

## Metrics

The model was evaluated using precision, recall, and F1 score.

Overall model performance on the test set:
- Precision: 0.7376
- Recall: 0.6404
- F1 Score: 0.6856

## Ethical Considerations

The dataset contains sensitive demographic information such as race, sex, and native
country. These features may introduce bias into the model predictions. The model should
not be used to make decisions that could discriminate against individuals based on these
protected characteristics. Users should be aware that historical census data may reflect
societal biases that could be perpetuated by the model.

## Caveats and Recommendations

- The model was trained on data from the 1994 U.S. Census and may not reflect current
  income distributions or demographics.
- The model should not be used as the sole basis for any financial or employment
  decisions.
- Further hyperparameter tuning and feature engineering could improve model performance.
- It is recommended to retrain the model periodically with more recent data.