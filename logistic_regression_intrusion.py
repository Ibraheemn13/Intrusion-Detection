from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from scipy.sparse import issparse
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend
import matplotlib.pyplot as plt




def main():
    Reg_Type = {
    }
    
    # fetch dataset
    rt_iot2022 = fetch_ucirepo(id=942)
    # data (as pandas dataframes)
    X = rt_iot2022.data.features
    y = rt_iot2022.data.targets


    X_sample = X.sample(frac=0.1, random_state=42)  # Adjusting  `frac` to 0.1
    y_sample = y.loc[X_sample.index]  # Ensuring corresponding targets are selected
    onehot = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_transformed = onehot.fit_transform(X_sample)

# Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_sample, test_size=0.25, random_state=42)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


# ----- No Regularization -----
# Initialize and fit the classifier
    print("\n\n----------- Logistic regression with no regularization ----------\n\n")
    model = LogisticRegression(penalty=None,solver='newton-cg',max_iter=1000)
    model.fit(X_train, y_train)

# Predict and evaluate
    y_pred = model.predict(X_test)
    print("\tResults with solver : newton-cg\n")
    print("Predictions:", y_pred)

    print("Classification Report: \n", classification_report(y_test, y_pred))
    print(f"The accuracy obtained: {accuracy_score(y_test, y_pred) * 100:.2f} %")

    Reg_Type["none"] = f"{accuracy_score(y_test, y_pred) * 100:.2f}"




# ----- L2 regularization (ridge) -----
# Initialize and fit the classifier
    print("\n\n----------- Logistic regression with l2 regularization ----------\n\n")
    model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)  # Increasing max_iter for convergence if needed
    model.fit(X_train, y_train)

# Predict and evaluate
    y_pred = model.predict(X_test)
    print("\tResults with solver : lbfgs\n")
    print("Predictions:", y_pred)


    print("Classification Report: \n", classification_report(y_test, y_pred))
    print(f"The accuracy obtained: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    Reg_Type["L2"] = f"{accuracy_score(y_test, y_pred) * 100:.2f}"



# ----- L1 regularization (Lasso) -----
# Initialize and fit the classifier
    print("\n\n----------- Logistic regression with l1 regularization ----------\n\n")
    model = LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000)  # Increasing max_iter for convergence if needed
    model.fit(X_train, y_train)

# Predict and evaluate
    y_pred = model.predict(X_test)
    print("\tResults with solver : liblinear\n")
    print("Predictions:", y_pred)


    print("Classification Report: \n", classification_report(y_test, y_pred))
    print(f"The accuracy obtained: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    Reg_Type["L1"] = f"{accuracy_score(y_test, y_pred) * 100:.2f}"


# ----- L1-L2 regularization (elastic-net) -----
# Initialize and fit the classifier
    print("\n\n----------- Logistic regression with l1-l2 regularization ----------\n\n")
    model = LogisticRegression(penalty='elasticnet',solver='saga',max_iter=1000, l1_ratio=0.5 )  # Increasing max_iter for convergence if needed
    model.fit(X_train, y_train)

# Predict and evaluate
    y_pred = model.predict(X_test)
    print("\tResults with solver : saga\n")
    print("Predictions:", y_pred)

    print("Classification Report: \n", classification_report(y_test, y_pred))
    print(f"The accuracy obtained: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    Reg_Type["L1-L2"] = f"{accuracy_score(y_test, y_pred) * 100:.2f}"



# ----- PCA dimensionality reduction -----
# OneHotEncoding the features
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_transformed = onehot.fit_transform(X_sample)

# Initialize IncrementalPCA
    n_components = 50  # Adjust as necessary
    ipca = IncrementalPCA(n_components=n_components)

# Fit IncrementalPCA on batches
    batch_size = 500  # Adjust based on memory capacity
    for i in range(0, X_transformed.shape[0], batch_size):
        ipca.partial_fit(X_transformed[i:i + batch_size])

# Transform the data in batches to avoid memory overload
    X_pca = np.concatenate([ipca.transform(X_transformed[i:i + batch_size]) for i in range(0, X_transformed.shape[0], batch_size)])

# Split the transformed data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_sample, test_size=0.25, random_state=42)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

# Initialize and fit the logistic regression model
    model = LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000)
    model.fit(X_train, y_train)

# Prediction and evaluation
    y_pred = model.predict(X_test)
    print("\n----------- PCA with No Regularization -----------\n")
    print("Predictions:", y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"The accuracy obtained: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    Reg_Type["PCA"] = f"{accuracy_score(y_test, y_pred) * 100:.2f}"
    
 
# ----- GRAPH -----
    print("\n----------- GRAPH displayed on seprate window -----------\n")
# Data preparation
    names = list(Reg_Type.keys())
    values = list(Reg_Type.values())

# Create bar graph
    plt.figure(figsize=(10, 5))  # Set the figure size as needed
    plt.bar(names, values, color='skyblue')  # You can choose different colors

# Adding title and labels
    plt.title('Model Accuracy by Regularization Type')
    plt.xlabel('Regularization Type')
    plt.ylabel('Accuracy (%)')
    plt.get_current_fig_manager().set_window_title('Model Accuracy by Regularization Type')

# Show the graph
    plt.show()


if __name__ == "__main__":
    main()
