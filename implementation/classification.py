"""
This script uses the features extracted using feature_extraction.py to
classify the audio samples into the 10 different genres.
"""
#%%
import os
import numpy as np
from sklearn import model_selection, utils, metrics
import pandas as pd
import sklearn.svm as SVM
pd.options.mode.chained_assignment = None

#%% Loading the previously extracted features and splitting features/labels
DF = pd.read_csv(os.path.dirname(__file__) + "/extracted_features.csv",
                 index_col=0)
print(DF.head())
X = DF.drop(columns='Genre')

GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop',
    'reggae', 'rock'
]
Y = DF['Genre'].apply(GENRES.index)


#%% Reducing dimensionality
def svd_dim_reduction(matrix, dims_to_remove):
    """
    Performs dimensionality reduction of a given matrix using SVD,
    removing the input number of dimensions.
    """
    if dims_to_remove == 0:
        return matrix
    U, S, V = np.linalg.svd(matrix)
    S = np.diag(S)
    dims_to_keep = len(S) - dims_to_remove
    U = U[:, 0:dims_to_keep]
    S = S[0:dims_to_keep, 0:dims_to_keep]
    V = np.transpose(V)[0:dims_to_keep]
    return np.dot(U, np.dot(S, V))


# Several tests have shown that reducing dimensionality does not help
# improve the accuracy of our classifier. Therefore we set dims_to_remove = 0.
X = svd_dim_reduction(X, 0)

#%% Training our classifier
# Shuffling the dataset so that the samples are not grouped by genre anymore.
X_SHUFFLED, Y_SHUFFLED = utils.shuffle(X, Y)

# We use a RBF kernel SVM as our classifier.
SVM_CLF = SVM.SVC(C=10, kernel='rbf', gamma='auto')

# Evaluating the classifier with a 10-fold cross validation
Y_PRED = model_selection.cross_val_predict(SVM_CLF,
                                           X_SHUFFLED,
                                           Y_SHUFFLED,
                                           cv=10)
#%% Recombining the 8 subsamples for each sample to get the final class
X_SHUFFLED['True genre'] = Y_SHUFFLED
X_SHUFFLED['Pred genre'] = Y_PRED

RESULT_DF = X_SHUFFLED[['True genre', 'Pred genre']]

# removing the part after '_' in each subsample name and grouping them
RESULT_DF['Original sample'] = RESULT_DF.index.to_series().apply(
    lambda s: s.split('_')[0])

# keeping the most frequent class among the 8 subsamples for each sample
GROUPED_RESULTS = RESULT_DF.groupby('Original sample').agg(
    lambda df: df.value_counts().index[0])

#%% Printing metrics : accuracy and confusion matrix
print("Accuracy on cross-validation : " + str(
    metrics.accuracy_score(GROUPED_RESULTS['True genre'],
                           GROUPED_RESULTS['Pred genre'])))

CONFUSION_MATRIX = pd.DataFrame(metrics.confusion_matrix(
    GROUPED_RESULTS['True genre'], GROUPED_RESULTS['Pred genre']),
                                index=GENRES,
                                columns=GENRES)
print("Confusion matrix :\n" + str(CONFUSION_MATRIX))
