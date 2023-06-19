import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import StackingClassifier

def train_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

def vectorize_and_train_model(model, X_train, y_train, X_test, y_test):
    text_clf = Pipeline([('tf', TfidfVectorizer(sublinear_tf= True, 
                       min_df = 5, 
                       norm= 'l2', 
                       ngram_range= (1,2), 
                       stop_words ='english') ),
                 ('clf', model)])
    train_predict(text_clf, X_train, y_train, X_test, y_test)

df = pd.read_csv("Consumer_Complaints.csv")
df = df[['Product', 'Consumer Complaint']]
df = df[pd.notnull(df['Consumer Complaint'])]
df.columns=['Product', 'Consumer_complaint']
df['category_id'] = df['Product'].factorize()[0]

fig = plt.figure(figsize= (8,6))
df.groupby('Product').Consumer_complaint.count().plot.bar(ylim=0)

X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint'], df['Product'], random_state= 0)

models = [LinearSVC(), BernoulliNB(), DecisionTreeClassifier(), 
          CatBoostClassifier(verbose=False), RandomForestClassifier(), XGBClassifier(), LGBMClassifier()]

for model in models:
    vectorize_and_train_model(model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning for RandomForestClassifier
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
               'min_samples_split': [2, 5, 10, 15, 100],
               'min_samples_leaf': [1, 2, 5, 10]}

rf = RandomForestClassifier()
rf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='accuracy', n_iter = 10, cv = 5, n_jobs=-1)
train_predict(rf, X_train_tfidf, y_train, X_test)

# Hyperparameter tuning for XGBClassifier
params = {"gamma": uniform(0, 0.5),
          "learning_rate": uniform(0.03, 0.3),
          "max_depth": randint(2, 6),
          "n_estimators": randint(100, 150),
          "subsample": uniform(0.6, 0.4)}

xgb = XGBClassifier()
xgb = RandomizedSearchCV(estimator = xgb, param_distributions = params, scoring='accuracy', n_iter = 10, cv = 5, n_jobs = -1)
train_predict(xgb, X_train_tfidf, y_train, X_test)

# Stacking Classifier
estimators = [('rf', RandomForestClassifier()), ('mb', BernoulliNB()), ('xgb', XGBClassifier())]
clf = StackingClassifier(estimators=estimators)
train_predict(clf, X_train_tfidf, y_train, X_test)
# For the RandomForest and XGBoost classifiers, the training has already been done above.
# So, we will just fetch the best estimators and use them to generate metrics.

# Evaluation metrics for RandomForestClassifier
best_rf = rf.best_estimator_
print("Best Parameters for RandomForestClassifier: ", rf.best_params_)
y_pred_rf = best_rf.predict(X_test)
print("Classification report for RandomForestClassifier:")
print(metrics.classification_report(y_test, y_pred_rf))

# Evaluation metrics for XGBClassifier
best_xgb = xgb.best_estimator_
print("Best Parameters for XGBClassifier: ", xgb.best_params_)
y_pred_xgb = best_xgb.predict(X_test)
print("Classification report for XGBClassifier:")
print(metrics.classification_report(y_test, y_pred_xgb))

# Evaluation metrics for Stacked Classifier
clf.fit(X_train_tfidf, y_train)
y_pred_clf = clf.predict(X_test)
print("Classification report for Stacked Classifier:")
print(metrics.classification_report(y_test, y_pred_clf))

# For visualization of the evaluations, let's plot the confusion matrices for all the classifiers
fig, axs = plt.subplots(3, 1, figsize=(10, 20))

plot_confusion_matrix(best_rf, X_test, y_test, ax=axs[0], cmap='Blues')
axs[0].set_title("Confusion Matrix for RandomForestClassifier")

plot_confusion_matrix(best_xgb, X_test, y_test, ax=axs[1], cmap='Blues')
axs[1].set_title("Confusion Matrix for XGBClassifier")

plot_confusion_matrix(clf, X_test, y_test, ax=axs[2], cmap='Blues')
axs[2].set_title("Confusion Matrix for Stacked Classifier")

plt.tight_layout()
plt.show()
