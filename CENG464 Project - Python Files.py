# -*- coding: utf-8 -*-


#--------------------- -----------------------Hasan's part -----------------------------------

#libraries
from IPython import get_ipython
from IPython.display import display
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


data = pd.read_excel("Data_processed.xlsx")


data['GrainYield'] = data['GrainYield'].str.strip()
data['GrainYield'] = data['GrainYield'].replace({'A': 0, 'B': 1, 'C': 2}) # target value(GrainYield) transforming a->0 , b-> 1, c->2


numeric_columns = data.select_dtypes(include=['number']).columns
selected_columns = list(numeric_columns)

# fill missing values with mean
for col in selected_columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col] = data[col].fillna(data[col].mean())
    else:
        data[col] = data[col].fillna(0)

#input x , target y
X = data[selected_columns].drop(columns=['GrainYield'])
y = data['GrainYield']


scaler = StandardScaler()           # scale the data
X = scaler.fit_transform(X)

# feature selection with rfe method
model = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
selected_features = [col for col, rank in zip(data[selected_columns].columns, rfe.ranking_) if rank == 1]

print("Selected Features:", selected_features)



# classification algorithms
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "ANN": MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', random_state=42, max_iter=500,
                        early_stopping=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# 5 fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def calculate_and_return_metrics(X_data, feature_name):  # calculates the performances of algorithms
    print(f"\nPerformance Comparison ({feature_name}):")
    results = []

    if feature_name == "Selected Features":
        print(f"Selected Features: {selected_features}\n")

    for name, clf in classifiers.items():
        scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        mcc_scores = []

        for train_idx, test_idx in skf.split(X_data, y):
            clf.fit(X_data[train_idx], y[train_idx])
            y_pred = clf.predict(X_data[test_idx])

            scores.append(accuracy_score(y[test_idx], y_pred))
            f1_scores.append(f1_score(y[test_idx], y_pred, average='weighted'))
            precision_scores.append(precision_score(y[test_idx], y_pred, average='weighted', zero_division=1))
            recall_scores.append(recall_score(y[test_idx], y_pred, average='weighted'))
            mcc_scores.append(matthews_corrcoef(y[test_idx], y_pred))

        # 5-fold results
        results.append([name,
                        np.mean(scores),
                        np.std(scores),
                        np.mean(f1_scores),
                        np.mean(precision_scores),
                        np.mean(recall_scores),
                        np.mean(mcc_scores)])

    # create and return DataFrame
    df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'std. deviation', 'F1', 'Precision', 'Recall', 'MCC'])
    display(df)
    return df


calculate_and_return_metrics(X, "All Features")


calculate_and_return_metrics(X_rfe, "Selected Features")


# confusion matrix for best model
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred = np.zeros_like(y)

for train_idx, test_idx in skf.split(X, y):
    best_model.fit(X[train_idx], y[train_idx])
    y_pred[test_idx] = best_model.predict(X[test_idx])

cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B", "C"])
cmd.plot(cmap="Blues")
plt.title("Confusion Matrix for Random Forest")
plt.show()

# ROC curve and AUC
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
classes = np.unique(y)
n_classes = len(classes)

for (name, clf), color in zip(classifiers.items(), colors):
    if hasattr(clf, "predict_proba"):
        y_score = np.zeros((len(y), n_classes))
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            y_score[test_idx] = clf.predict_proba(X[test_idx])

        fpr, tpr, _ = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(label_binarize(y, classes=classes)[:, i], y_score[:, i])

        # avg ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        mean_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.2f})', color=color)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Classifiers')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# excel output part
import openpyxl
df_all = calculate_and_return_metrics(X, "All Features")
df_selected = calculate_and_return_metrics(X_rfe, "Selected Features")


results_all_features = df_all.values.tolist()
results_selected_features = df_selected.values.tolist()
workbook = openpyxl.Workbook()

sheet_all = workbook.active
sheet_all.title = "All Features"
sheet_all.append(["Classifier", "Accuracy", "std. deviation", "F1", "Precision", "Recall", "MCC"])
for row in results_all_features:
    sheet_all.append(row)


sheet_selected = workbook.create_sheet(title="Selected Features")
sheet_selected.append(["Selected Features"])
for feature in selected_features:
    sheet_selected.append([feature])

sheet_selected.append([])
sheet_selected.append(["Classifier", "Accuracy", "std. deviation", "F1", "Precision", "Recall", "MCC"])
for row in results_selected_features:
    sheet_selected.append(row)

# Save as hasan_data.xlsx
file_name = "hasan_data.xlsx"
workbook.save(file_name)

print(f"Excel file saved as {file_name}")

# ------------------------------------------------------- Gökay's part -------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.ensemble._weight_boosting")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")



def clean_data(data): #convert columns numeric and removing special characters
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = data[column].replace(r'[^\d.]+', '', regex=True)
        data[column] = pd.to_numeric(data[column], errors='coerce')
    return data


data = pd.read_excel("Data_processed.xlsx")


data['GrainYield'] = data['GrainYield'].str.strip()
data['GrainYield'] = data['GrainYield'].replace({'A': 0, 'B': 1, 'C': 2}) # target(GrainYield) convert a=0 , b=1, c=2


data = clean_data(data)


for col in data.columns: # fill missing data with mean
    if data[col].dtype in ['float64', 'int64']:
        data[col] = data[col].fillna(data[col].mean())
    else:
        data[col] = data[col].fillna(0)


X = data.drop(columns=['GrainYield']) # input x, target y
y = data['GrainYield']

# split the data  training and test sets (%20 test, %80 training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


selector = SelectKBest(score_func=f_classif, k=2) # feature selection (SelectKBest with the best 2 features)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected Features: {selected_features}")

# classification algorithms
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000, solver='saga'),
    "ANN": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}


def evaluate_model(name, clf, X_train, X_test, y_train, y_test, label): #performances of algorithms
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # calculate AUC
    if y_proba is not None:
        if y_proba.shape[1] > 1:
            auc = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = 'N/A'


    print(f"\n{name} Model Results ({label}):")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc}")


    return {
        'Model': name,
        'Label': label,
        'Accuracy': accuracy * 100,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc,
        'AUC': auc
}




print("\nResults using All Features:")
for name, clf in classifiers.items(): # evaluate model with all features
    evaluate_model(name, clf, X_train_scaled, X_test_scaled, y_train, y_test, label="All Features")

print("\nResults using Selected Features:")
for name, clf in classifiers.items(): # evaluate model with selected features
    evaluate_model(name, clf, X_train_selected, X_test_selected, y_train, y_test, label="Selected Features")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(name, clf, X_test, y_test, label): # ROC Curve plotting function
    # Get predicted probabilities for the test data
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    if y_proba is not None:
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)  # For binary classification
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# plot ROC curve for all classifiers
plt.figure(figsize=(10, 8))
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    plot_roc_curve(name, clf, X_test_scaled, y_test, label="All Features")

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve for All Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


classes = np.unique(y)


model_scores = {} # dictionary to select the best algorithm

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_scores[name] = accuracy


best_model_name = max(model_scores, key=model_scores.get) # select the best algorithm
best_model = classifiers[best_model_name]

# Train and test the best model
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for {best_model_name}")
plt.show()

# print performance metrics
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, average='weighted')
recall = recall_score(y_test, y_pred_best, average='weighted')
f1 = f1_score(y_test, y_pred_best, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred_best)

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")

# evaluate All Features results to select the best model
best_model_all_features = max(results_all_features, key=lambda x: x[1])  # Select by highest accuracy value
best_model_name_all_features = best_model_all_features[0]
best_model_metrics_all_features = best_model_all_features[1:]  # Accuracy, Precision, Recall, F1 Score, MCC, AUC

#excel output part
import pandas as pd

# empty lists to store results
all_features_results = []
selected_features_results = []


for name, clf in classifiers.items(): # evaluate models All Features
    result = evaluate_model(name, clf, X_train_scaled, X_test_scaled, y_train, y_test, label="All Features")
    all_features_results.append(result)


for name, clf in classifiers.items():  # evaluate models  Selected Features
    result = evaluate_model(name, clf, X_train_selected, X_test_selected, y_train, y_test, label="Selected Features")
    selected_features_results.append(result)

# convert dataframe
df_all_features = pd.DataFrame(all_features_results)
df_selected_features = pd.DataFrame(selected_features_results)


with pd.ExcelWriter('gokay_data.xlsx') as writer: # save as gokay_data.xlsx
    df_all_features.to_excel(writer, sheet_name='All Features', index=False)
    df_selected_features.to_excel(writer, sheet_name='Selected Features', index=False)

print("Results saved to 'gokay_data.xlsx' file.")

import pandas as pd

#read "hasan_data.xlxs" and "gokay_data.xlsx"
hasan_data = pd.read_excel('hasan_data.xlsx', sheet_name=None)
gokay_data = pd.read_excel('gokay_data.xlsx', sheet_name=None)


with pd.ExcelWriter('data.xlsx') as writer: #merge data into "data.xlsx" file

    for sheet_name, data in hasan_data.items():
        data.to_excel(writer, sheet_name=f'Hasan_{sheet_name}', index=False)

    for sheet_name, data in gokay_data.items():
        data.to_excel(writer, sheet_name=f'Gokay_{sheet_name}', index=False)

print("Data successfully saved to 'data.xlsx' file.")