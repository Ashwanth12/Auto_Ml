import os
import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components
from pivottablejs import pivot_ui
import numpy as np
from scikitplot import metrics
from scipy.stats import zscore
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MiniBatchKMeans, \
    MeanShift, OPTICS, SpectralClustering
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, \
    ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from streamlit_pandas_profiling import st_profile_report

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Cleaning", "Profiling", "Data visualization", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        columns = df.columns
        st.subheader("Shape and size of the data")
        if st.button("size"):
            st.write(df.shape)
            st.write(df.size)
        st.subheader("Do you want description about data ")
        if st.button("Yes", key='describe'):
            st.dataframe(df.describe())
        st.subheader("Do you want information about data ")
        if st.button("Yes", key='info'):
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        st.subheader("Check whether data has duplicates")
        if st.button("Yes", key='duplicate'):
            st.write("Number of duplicate rows: ", df.duplicated().sum())
        st.subheader("Check whether data has missing values")
        if st.button("Yes", key="missing"):
            st.write("Number of missing rows")
            st.dataframe(df.isnull().sum())

if choice == "Cleaning":
    remove_cols = st.multiselect("Do you want to remove any columns", df.columns)
    ava_col = []
    if remove_cols:
        df.drop(remove_cols, axis=1, inplace=True)
    if st.button("Show Dataset"):
        st.dataframe(df.head(5))
        st.write(df.columns)

    st.write("Do you want to remove duplicates?")
    if st.button("Yes", key='remove duplicate'):
        df.drop_duplicates(inplace=True)
        st.write("Number of duplicate rows: ", df.duplicated().sum())
    filling_col = st.multiselect("Select columns to fill missing values", df.columns)

    if len(filling_col) > 0:
        for column in filling_col:
            dt = str(df[column].dtype)

            if dt == 'int64' or dt == 'float64':
                option = st.radio(
                    "Fill missing values with",
                    ("Constant", "Mean", "Median"),
                    key=f"fill_option_{column}",
                    help="Choose the method to fill missing numeric values"
                )
            elif dt == 'object':
                option = st.radio(
                    "Fill missing values with",
                    ("Constant", "Mode"),
                    key=f"fill_option_{column}",
                    help="Choose the method to fill missing categorical values"
                )

            if option == "Constant":
                con = st.text_input("Enter the filling constant")
                df.fillna(con, inplace=True)
            elif option == 'Mean':
                df.fillna(df[column].mean(), inplace=True)
            elif option == 'Median':
                df.fillna(df[column].median(), inplace=True)
            elif option == 'Mode':
                df.fillna(df[column].mode()[0], inplace=True)
            st.write(df)

    outlier_columns = st.multiselect("Select columns for outlier detection", df.columns)

    if len(outlier_columns) > 0:
        for column in outlier_columns:
            option = st.selectbox(
                "Outliers using",
                ('Z-Score', 'IQR', 'Percentile'),
                help="Choose the method for outlier detection"
            )

            if option == "Z-Score":
                trimming_option = st.radio(
                    "Outlier detection method",
                    ("Trimming", "Capping"),
                    key=f"zscore_option_{column}",
                    help="Choose the method for outlier detection"
                )

                if trimming_option == "Trimming":
                    z_scores = zscore(df[column])
                    df = df[(np.abs(z_scores) < 3)]
                elif trimming_option == "Capping":
                    z_scores = zscore(df[column])
                    lower_threshold = -3
                    upper_threshold = 3
                    df[column] = np.where(df[column] < lower_threshold, lower_threshold, df[column])
                    df[column] = np.where(df[column] > upper_threshold, upper_threshold, df[column])

            elif option == "IQR":
                option = st.radio(
                    "Outlier detection method",
                    ("Trimming", "Capping"),
                    key=f"iqr_option_{column}",
                    help="Choose the method for outlier detection"
                )
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_threshold = q1 - 1.5 * iqr
                upper_threshold = q3 + 1.5 * iqr

                if option == "Trimming":
                    df = df[df[column].between(lower_threshold, upper_threshold)]
                elif option == "Capping":
                    df[column] = np.where(df[column] < lower_threshold, lower_threshold, df[column])
                    df[column] = np.where(df[column] > upper_threshold, upper_threshold, df[column])

            elif option == "Percentile":
                option = st.radio(
                    "Outlier detection method",
                    ("Trimming", "Capping"),
                    key=f"percentile_option_{column}",
                    help="Choose the method for outlier detection"
                )
                lower_percentile = st.number_input("Enter the lower percentile (0-50)", value=5, min_value=0,
                                                   max_value=50)
                upper_percentile = st.number_input("Enter the upper percentile (50-100)", value=95, min_value=50,
                                                   max_value=100)
                lower_limit = df[column].quantile(lower_percentile / 100)
                upper_limit = df[column].quantile(upper_percentile / 100)

                if option == "Trimming":
                    df = df[df[column].between(lower_limit, upper_limit)]
                elif option == "Capping":
                    df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
                    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
            st.write(df)
    df.to_csv("data.csv")

if choice == "Data visualization":
    if not os.path.exists('./data.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    df = pd.read_csv("./data.csv")
    df = df.iloc[:, 1:]
    st.title("Pivot Table")
    t = pivot_ui(df)
    with open(t.src) as t:
        components.html(t.read(), width=900, height=1000, scrolling=True)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile = ProfileReport(df)
    st_profile_report(profile)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    target = st.slider('Test_Size', 0.01, 0.5)
    random_state = st.slider('Random_State', 0, 100)
    choice1 = st.radio("Machine Learning Models", ["Regression", "Classification", "Clustering"])

    X = df.drop(columns=[chosen_target])
    y = df[chosen_target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=target, random_state=random_state)
    numerical_col = X.select_dtypes(include=np.number).columns
    categorical_col = X.select_dtypes(exclude=np.number).columns
    scaler = StandardScaler()
    ct_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_col)],
                                   remainder='passthrough')
    x_train_encoded = ct_encoder.fit_transform(x_train)
    x_test_encoded = ct_encoder.transform(x_test)
    x_train = scaler.fit_transform(x_train_encoded)
    x_test = scaler.transform(x_test_encoded)

    if choice1 == "Regression":
        algorithms = st.multiselect("Regression Algorithms",
                                    ["Linear Regression", "Polynomial Regression",
                                     "Support Vector Regression", "Decision Tree Regression",
                                     "Random Forest Regression", "Ridge Regression",
                                     "Lasso Regression", "Gaussian Regression", "KNN Regression", "AdaBoost"])

        if st.button('Run Modelling'):
            table = {"Algorithm": [], "MAE": [], "RMSE": [], "R2 Score": []}
            for algorithm in algorithms:
                if algorithm == "Linear Regression":
                    reg = LinearRegression()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Linear Regression")
                    table["R2 Score"].append(r2score)

                elif algorithm == "Polynomial Regression":
                    degree = st.slider("Polynomial Degree", 2, 10, 2)
                    reg = PolynomialFeatures(degree=degree)
                    x_train_poly = reg.fit_transform(x_train)
                    x_test_poly = reg.transform(x_test)
                    model = LinearRegression()
                    model.fit(x_train_poly, y_train)
                    y_pred = model.predict(x_test_poly)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Polynomial Regression (Degree: {})".format(degree))
                    table["R2 Score"].append(r2score)

                elif algorithm == "Support Vector Regression":
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1)
                    reg = SVR(kernel=kernel, epsilon=epsilon)
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append(
                        "Support Vector Regression (Kernel: {}, Epsilon: {})".format(kernel, epsilon))
                    table["R2 Score"].append(r2score)

                elif algorithm == "Decision Tree Regression":
                    reg = DecisionTreeRegressor()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Decision Tree Regression")
                    table["R2 Score"].append(r2score)

                elif algorithm == "Random Forest Regression":
                    n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                    reg = RandomForestRegressor(n_estimators=n_estimators)
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Random Forest Regression (Estimators: {})".format(n_estimators))
                    table["R2 Score"].append(r2score)

                elif algorithm == "Ridge Regression":
                    alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
                    reg = Ridge(alpha=alpha)
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Ridge Regression (Alpha: {})".format(alpha))
                    table["R2 Score"].append(r2score)

                elif algorithm == "Lasso Regression":
                    alpha = st.slider("Alpha1", 0.01, 10.0, 1.0)
                    reg = Lasso(alpha=alpha)  # Fix: Initialize Lasso Regression model
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Lasso Regression (Alpha: {})".format(alpha))
                    table["R2 Score"].append(r2score)

                elif algorithm == "Gaussian Regression":
                    reg = GaussianProcessRegressor()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Gaussian Regression")
                    table["R2 Score"].append(r2score)

                elif algorithm == "AdaBoost":
                    n_estimators = st.slider("Number of Estimators for AdaBoost", 10, 200, 100)
                    reg = AdaBoostRegressor(n_estimators=n_estimators)
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("AdaBoost")
                    table["R2 Score"].append(r2score)

            df_results = pd.DataFrame(table)
            st.write(df_results)
            st.snow()

    elif choice1 == "Classification":
        algorithms = st.multiselect("Classification Algorithms", ["Logistic Regression", "Decision Trees",
                                                                  "Random Forest", "Naive Bayes",
                                                                  "Support Vector Machines (SVM)",
                                                                  "Gradient Boosting", "Neural Networks",
                                                                  "Quadratic Discriminant Analysis (QDA)",
                                                                  "Adaptive Boosting (AdaBoost)",
                                                                  "Gaussian Processes", "Hidden Markov Models (HMM)",
                                                                  "DecisionTreeClassifier", "Perceptron",
                                                                  "Ridge Classifier", "Passive Aggressive Classifier",
                                                                  "Elastic Net", "Lasso Regression", ])

        if st.button('Run Modelling'):
            table = {"Algorithm": [], "Precision": [], "Recall": [], "F1-Score": []}
            for algorithm in algorithms:
                if algorithm == "Logistic Regression":
                    reg = LogisticRegression()
                    reg.fit(x_train_encoded, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Logistic Regression")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Decision Trees":
                    reg = DecisionTreeClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Decision Trees")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Random Forest":
                    reg = RandomForestClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Random Forest")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Naive Bayes":
                    reg = GaussianNB()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Naive Bayes")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Support Vector Machines (SVM)":
                    reg = SVC()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Support Vector Machines (SVM)")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Gradient Boosting":
                    reg = GradientBoostingClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Gradient Boosting")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Neural Networks":
                    reg = MLPClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Neural Networks")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Quadratic Discriminant Analysis (QDA)":
                    reg = QuadraticDiscriminantAnalysis()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Quadratic Discriminant Analysis (QDA)")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Adaptive Boosting (AdaBoost)":
                    reg = AdaBoostClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Adaptive Boosting (AdaBoost)")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Gaussian Processes":
                    reg = GaussianProcessClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Gaussian Processes")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Hidden Markov Models (HMM)":
                    from hmmlearn.hmm import GaussianHMM

                    reg = GaussianHMM()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Hidden Markov Models (HMM)")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "DecisionTreeClassifier":
                    reg = DecisionTreeClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("DecisionTreeClassifier")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Perceptron":
                    reg = Perceptron()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Perceptron")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Ridge Classifier":
                    reg = RidgeClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Ridge Classifier")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Passive Aggressive Classifier":
                    reg = PassiveAggressiveClassifier()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Passive Aggressive Classifier")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Elastic Net":
                    reg = ElasticNet()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Elastic Net")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)

                elif algorithm == "Lasso Regression":
                    reg = Lasso()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred) * 100
                    precision = precision_score(y_test, y_pred, average='weighted') * 100
                    recall = recall_score(y_test, y_pred, average='weighted') * 100
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100

                    table["Algorithm"].append("Lasso Regression")
                    table["Precision"].append(precision)
                    table["Recall"].append(recall)
                    table["F1-Score"].append(f1)
            df_results = pd.DataFrame(table)
            st.write(df_results)
            st.snow()

    elif choice1 == "Clustering":
        algorithms = st.multiselect("Clustering Algorithms", ["Affinity Propagation", "Agglomerative Clustering",
                                                              "BIRCH", "DBSCAN", "K-Means", "Mini-Batch K-Means",
                                                              "Mean Shift", "OPTICS", "Spectral Clustering",
                                                              "Gaussian Mixture Model"])

        if st.button('Run Modelling'):
            table = {"Algorithm": [], "Accuracy": []}
            for algorithm in algorithms:
                if algorithm == "Affinity Propagation":
                    clustering = AffinityPropagation()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "Agglomerative Clustering":
                    clustering = AgglomerativeClustering()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "BIRCH":
                    clustering = Birch()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "DBSCAN":
                    clustering = DBSCAN()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "K-Means":
                    clustering = KMeans()
                    clustering.fit(x_train)
                    labels = clustering.predict(x_train)
                    silhouette_avg = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append("K-Means")
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "Mini-Batch K-Means":
                    clustering = MiniBatchKMeans()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "Mean Shift":
                    clustering = MeanShift()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "OPTICS":
                    clustering = OPTICS()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "Spectral Clustering":
                    clustering = SpectralClustering()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))

                elif algorithm == "Gaussian Mixture Model":
                    clustering = GaussianMixture()
                    clustering.fit(x_train)
                    labels = clustering.labels_
                    silhouette_score = metrics.silhouette_score(x_train, labels)
                    table["Algorithm"].append(algorithm)
                    table["Accuracy"].append(metrics.silhouette_score(x_train, labels))
            st.write(pd.DataFrame(table))
            st.snow()
