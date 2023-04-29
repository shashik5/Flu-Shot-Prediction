import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from feature_selection import select_features
from pre_process import pre_process, pre_process_using_pandas


def get_predicted_probability(trainingDataset: np.array, trainingLabels: np.array, testingDataset: np.array, selectedFeatures: pd.Series):
    idx = np.array(selectedFeatures)
    x = trainingDataset[:, idx]
    params_to_test = {
        'n_estimators': [50, 100, 500, 1000, 1500],
        'max_depth': [10, 20, 30, 40]
        # 'C': [1, 10, 20],
        # 'kernel': ['rbf']
    }
    model = GridSearchCV(RandomForestClassifier(), param_grid=params_to_test, cv=5, return_train_score=False, n_jobs=-1)
    # model = GridSearchCV(svm.SVC(gamma='auto', probability=True), param_grid=params_to_test, cv=5, return_train_score=False, n_jobs=-1)
    model.fit(x, trainingLabels)
    print(model.best_score_)
    y_prob = model.predict(x)
    print(roc_auc_score(trainingLabels, y_prob))
    return model.predict_proba(testingDataset[:, idx])[:, 1].reshape(-1, 1)


def get_predicted_probability_using_ANN(trainingDataset: np.array, trainingLabels: np.array, testingDataset: np.array):
    x = trainingDataset[:, :]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='selu'))
    model.add(tf.keras.layers.Dense(units=32, activation='selu'))
    model.add(tf.keras.layers.Dense(units=16, activation='selu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(x, trainingLabels, batch_size = 1024, epochs = 1000)
    y_prob = model.predict(x)
    print(roc_auc_score(trainingLabels, np.round(y_prob)))
    result = model.predict(testingDataset[:, :])
    return result


training_set, training_labels, testing_set = pre_process(trainingDataFilePath='./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv',
     trainingLabelFilePath='./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv', testingDataFilePath='./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Test_Features.csv')

# training_set, training_labels, testing_set = pre_process_using_pandas(trainingDataFilePath='./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv',
#                                                                       trainingLabelFilePath='./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv', testingDataFilePath='./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Test_Features.csv')

print(training_set.shape)
print(training_labels.shape)
assert training_set.shape[0] == training_labels.shape[0]

# Using Random Forest and SVC methods
h1n1_features, seasonal_features = select_features(training_set, training_labels)
y_pred_prob_h1n1 = get_predicted_probability(training_set, training_labels[:, 0], testing_set, h1n1_features)
y_pred_prob_seasonal = get_predicted_probability(training_set, training_labels[:, 1], testing_set, seasonal_features)

# Using ANN
# y_pred_prob_h1n1 = get_predicted_probability_using_ANN(training_set, training_labels[:, 0], testing_set)
# y_pred_prob_seasonal = get_predicted_probability_using_ANN(training_set, training_labels[:, 1], testing_set)

submission_dataset = pd.read_csv('./resources/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Submission_Format.csv', index_col="respondent_id")
submission_dataset.head()

submission_dataset["h1n1_vaccine"] = y_pred_prob_h1n1
submission_dataset["seasonal_vaccine"] = y_pred_prob_seasonal
submission_dataset.head()

date = pd.Timestamp.now().strftime(format='%Y-%m-%d_%H-%M_')
submission_dataset.to_csv(f'./submissions/my_submission_{date}.csv', index=True)
