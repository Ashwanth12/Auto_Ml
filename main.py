import os
import pickle
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from pivottablejs import pivot_ui
import numpy as np
# from scikitplot import metrics
from scipy.stats import zscore
import io
import pygwalker as pyg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap
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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation",
                      ["Introduction", "Upload", "Cleaning", "Profiling", "Data visualization", "Modelling",
                       "Download"])
    st.info("This project application helps you build and explore your data.")

    components.html("""
    <script async <!-- Global site tag (gtag.js) Google Analytics -->
    src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5056338602918094" crossorigin="anonymous"></script>
    < script>
    window.dataLayer = window.dataLayer || []; function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());""")
if choice == "Introduction":
    st.write("# Welcome to machine learning project platform! 👋")
    st.markdown("""
    At our website, we offer a comprehensive suite of tools and features to assist you in your data-driven projects. Whether you're a data enthusiast, a business analyst, or a machine learning practitioner, our platform is designed to streamline your workflow and help you make insightful decisions from your data.
    
    ### *Upload:*
    In this section, you can effortlessly upload your dataset and explore its contents. Once your data is uploaded, our platform provides a detailed description of the dataset, highlighting key statistics, such as the number of rows, columns, and unique values. You can quickly identify potential data quality issues by checking for the presence of duplicate records and missing values in each column.
    """)
    st.image(Image.open("images/upload.png"))
    st.markdown("""
    ### *Cleaning:*
    The cleaning section empowers you to enhance the quality of your data effortlessly. If there are any unnecessary columns in your dataset, you can easily remove them to focus on the relevant information. Additionally, you have the option to remove duplicates and handle missing values intelligently. Our platform will assist you in maintaining a clean and accurate dataset for further analysis.""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(Image.open("images/Cleaning1.png"))

    with col2:
        st.image(Image.open("images/Cleaning2.png"))

    with col3:
        st.image(Image.open("images/Cleaning3.png"))
    st.markdown("""
    ### *Profiling:*
    Our profiling feature leverages the power of pandas profiling to provide you with a comprehensive overview of your dataset. You will gain insights into data types, descriptive statistics, and correlations between variables. With this detailed meta-information, you can better understand the underlying patterns and relationships within your data.""")
    st.image(Image.open("images/profiling.png"))
    st.markdown("""
    ### *Data Visualization:*
    Data visualization is key to uncovering hidden patterns and presenting insights effectively. In this section, we offer a rich set of visualizations, akin to the capabilities of popular tools like Tableau. Our interactive charts and plots will help you create compelling visuals to showcase your data's story, making it easier for your audience to grasp complex information at a glance.""")
    st.image(Image.open("images/pivot.png"))
    st.markdown("""
    ### *Modelling:*
    For the data scientists and machine learning enthusiasts, the modeling section is a treasure trove of possibilities. Begin by selecting the target variable and specifying the problem type - be it regression, classification, clustering, or dimensionality reduction. Then, fine-tune the hyperparameters of various algorithms to obtain the best results. Once the models are trained, our platform will present a detailed performance summary, including essential metrics such as MAE, RMSE, R2 score, precision, recall, and F1 score. You'll be empowered to make data-driven decisions with confidence.""")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.image(Image.open("images/model1.png"))

    with col5:
        st.image(Image.open("images/model2.png"))

    with col6:
        st.image(Image.open("images/model3.png"))

    st.markdown("""
    ### *Download:*
    At the end of your journey, you can seamlessly download your model in the form of convenient pickle files, enabling you to deploy your trained model easily. Additionally, if you need a copy of your processed dataset, you can download it in the universally-accepted CSV format, ensuring your data remains accessible for further analysis or sharing.""")
    st.image(Image.open("images/download.png"))
    st.markdown("""
    Join us in exploring the power of data analysis and machine learning! Our platform is your gateway to uncovering hidden insights, making data-driven decisions, and bringing your projects to new heights. Let's embark on this exciting journey together!""")
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        describe_table = df.describe()
        minmax = {}
        for i in describe_table:
            minmax[i] = [describe_table[i]['min'], describe_table[i]['max']]
        st.session_state['minmaxtable'] = minmax
        columns = df.columns
        st.subheader("Shape and size of the data")
        if st.button("size"):
            st.write(df.shape)
            st.write(df.size)
        st.subheader("Do you want description about data ")
        if st.button("Yes", key='describe'):
            st.dataframe(describe_table)
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
    if not os.path.exists('./dataset.csv'):
        st.subheader("Go To Upload File")
    else:
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
    if os.path.exists('./data.csv'):
        df = pd.read_csv("./data.csv")
        df = df.iloc[:, 1:]
        st.title("Data visualization")
        t = pivot_ui(df)
        with open(t.src) as t:
            components.html(t.read(), width=1200, height=1200, scrolling=True)
        pyg_html = pyg.to_html(df)
        components.html(pyg_html, height=1000, scrolling=True)
    else:
        if os.path.exists('./dataset.csv'):
            # df = pd.read_csv("./dataset.csv")
            df = df.iloc[:, 1:]
            st.title("Pivot Table")
            t = pivot_ui(df)
            with open(t.src) as t:
                components.html(t.read(), width=1200, height=1200, scrolling=True)
        else:
            st.subheader("GO To Upload File")

if choice == "Profiling":
    if os.path.exists('./data.csv'):
        df = pd.read_csv("./data.csv")
        df = df.iloc[:, 1:]
        st.title("Exploratory Data Analysis")
        profile = ProfileReport(df)
        st_profile_report(profile)
    else:
        if os.path.exists('./dataset.csv'):
            st.title("Exploratory Data Analysis")
            profile = ProfileReport(df)
            st_profile_report(profile)
        else:
            st.subheader("GO To Upload File")

if choice == "Modelling":
    snow = False
    df_results = []
    if not os.path.exists('./data.csv'):
        st.subheader("Go To Upload File")
    else:
        df = pd.read_csv("./data.csv")
        df = df.iloc[:, 1:]
        df_clone = df.iloc[:, 1:]
        col1, col2 = st.columns(2, gap='medium')
        with col1:
            st.subheader("supervised machine learning")
            choice1 = st.selectbox("supervised", ["Regression", "Classification"])
            chosen_target = st.selectbox('Choose the Target Column', df.columns, index=len(df.columns) - 1)
            st.session_state['chosen_target'] = chosen_target
            target = st.slider('Test_Size', 0.01, 0.5)
            random_state = st.slider('Random_State', 0, 100)
            X = df.drop(columns=[chosen_target])
            y = df[chosen_target]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=target, random_state=random_state)
            numerical_col = X.select_dtypes(include=np.number).columns
            st.session_state['numerical_col_set'] = set(numerical_col)
            categorical_col = X.select_dtypes(exclude=np.number).columns
            st.session_state['categorical_col_set'] = set(categorical_col)
            scaler = MinMaxScaler()
            ct_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_col)],
                                           remainder='passthrough')
            st.session_state['ct_encoder'] = ct_encoder
            x_train_encoded = ct_encoder.fit_transform(x_train)
            x_test_encoded = ct_encoder.transform(x_test)
            x_train = scaler.fit_transform(x_train_encoded)
            x_test = scaler.transform(x_test_encoded)

            if "Regression" in choice1:
                algorithms = st.multiselect("Regression Algorithms",
                                            ["Linear Regression", "Polynomial Regression",
                                             "Support Vector Regression", "Decision Tree Regression",
                                             "Random Forest Regression", "Ridge Regression",
                                             "Lasso Regression", "Gaussian Regression", "KNN Regression", "AdaBoost"])

                if True:
                    snow = True
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
                            pickle.dump(reg, open('LR.pkl', 'wb'))

                        elif algorithm == "Polynomial Regression":
                            degree = st.slider("Polynomial Degree", 2, 10, 2)
                            reg = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
                            reg.fit(x_train, y_train)
                            y_pred = reg.predict(x_test)
                            mse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            r2score = r2_score(y_test, y_pred) * 100

                            table["RMSE"].append(mse)
                            table["MAE"].append(mae)
                            table["Algorithm"].append("Polynomial Regression (Degree: {})".format(degree))
                            table["R2 Score"].append(r2score)
                            pickle.dump(reg, open('PR.pkl', 'wb'))

                        elif algorithm == "Support Vector Regression":
                            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                            epsilon = st.slider("Svm Epsilon", 0.01, 1.0, 0.1)
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
                            pickle.dump(reg, open('SVR.pkl', 'wb'))

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
                            pickle.dump(reg, open('DTR.pkl', 'wb'))

                        elif algorithm == "Random Forest Regression":
                            n_estimators = st.slider("Number of Estimators for Random Forest", 10, 200, 100)
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
                            pickle.dump(reg, open('RFR.pkl', 'wb'))

                        elif algorithm == "Ridge Regression":
                            alpha = st.slider("Ridge Alpha", 0.01, 10.0, 1.0)
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
                            pickle.dump(reg, open('RR.pkl', 'wb'))

                        elif algorithm == "Lasso Regression":
                            alpha = st.slider(" Lasso Alpha1", 0.01, 10.0, 1.0)
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
                            pickle.dump(reg, open('LASR.pkl', 'wb'))

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
                            pickle.dump(reg, open('GR.pkl', 'wb'))
                        elif algorithm == "KNN Regression":
                            k_neighbors = st.slider("Number of Neighbors (K) for KNN", 1, 20, 5)
                            reg = KNeighborsRegressor(n_neighbors=k_neighbors)
                            reg.fit(x_train, y_train)
                            y_pred = reg.predict(x_test)
                            mse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            r2score = r2_score(y_test, y_pred) * 100

                            table["RMSE"].append(mse)
                            table["MAE"].append(mae)
                            table["Algorithm"].append("K-Nearest Neighbors Regression (K={})".format(k_neighbors))
                            table["R2 Score"].append(r2score)
                            pickle.dump(reg, open('KNR.pkl', 'wb'))

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
                            pickle.dump(reg, open('ABR.pkl', 'wb'))
                    if st.button('Run Modelling'):
                        df_results = pd.DataFrame(table)

            elif "Classification" in choice1:
                label = {}
                classes = {}
                v = 0
                for i in y.unique():
                    label[i] = v
                    classes[v] = i
                    v += 1
                st.session_state['classes'] = classes
                y_test = y_test.apply(lambda x: label[x])
                y_train = y_train.apply(lambda x: label[x])

                algorithms = st.multiselect(
                    "Classification Algorithms", ["Logistic Regression", "Decision Trees", "Random Forest", "Naive Bayes",
                                                   "Support Vector Machines (SVM)", "Gradient Boosting", "Neural Networks",
                                                   "Quadratic Discriminant Analysis (QDA)", "Adaptive Boosting (AdaBoost)",
                                                   "Gaussian Processes", "Perceptron", "KNN Classifier", "Ridge Classifier",
                                                   "Passive Aggressive Classifier", "Elastic Net", "Lasso Regression"])
                if True:
                    snow = True
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
                            pickle.dump(reg, open('LOR.pkl', 'wb'))

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
                            pickle.dump(reg, open('DT.pkl', 'wb'))

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
                            pickle.dump(reg, open('RF.pkl', 'wb'))

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
                            pickle.dump(reg, open('NB.pkl', 'wb'))

                        elif algorithm == "Support Vector Machines (SVM)":

                            reg = SVC(decision_function_shape='ovo')
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
                            pickle.dump(reg, open('SVM.pkl', 'wb'))

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
                            pickle.dump(reg, open('GB.pkl', 'wb'))

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
                            pickle.dump(reg, open('NN.pkl', 'wb'))

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
                            pickle.dump(reg, open('QDA.pkl', 'wb'))

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
                            pickle.dump(reg, open('AB.pkl', 'wb'))

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
                            pickle.dump(reg, open('GP.pkl', 'wb'))

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
                            pickle.dump(reg, open('PT.pkl', 'wb'))

                        elif algorithm == "KNN Classifier":
                            k_neighbors = st.slider("Number of Neighbors for KNN", 1, 20, 5)
                            reg = KNeighborsClassifier(n_neighbors=k_neighbors)
                            reg.fit(x_train, y_train)
                            y_pred = reg.predict(x_test)
                            accuracy = accuracy_score(y_test, y_pred) * 100
                            precision = precision_score(y_test, y_pred, average='weighted') * 100
                            recall = recall_score(y_test, y_pred, average='weighted') * 100
                            f1 = f1_score(y_test, y_pred, average='weighted') * 100

                            table["Algorithm"].append("K-Nearest Neighbors Classifier (K={})".format(k_neighbors))
                            table["Precision"].append(precision)
                            table["Recall"].append(recall)
                            table["F1-Score"].append(f1)
                            pickle.dump(reg, open('KNC.pkl', 'wb'))

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
                            pickle.dump(reg, open('RC.pkl', 'wb'))

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
                            pickle.dump(reg, open('PA.pkl', 'wb'))

                        elif algorithm == "Elastic Net":
                            x_train = pd.DataFrame(x_train)
                            y_train = pd.DataFrame(y_train)
                            x_test = pd.DataFrame(x_test)
                            y_test = pd.DataFrame(y_test)

                            x_train = x_train.apply(pd.to_numeric, errors='coerce')
                            y_train = y_train.apply(pd.to_numeric, errors='coerce')
                            x_test = x_test.apply(pd.to_numeric, errors='coerce')
                            y_test = y_test.apply(pd.to_numeric, errors='coerce')
                            x_train.fillna(0, inplace=True)
                            y_train.fillna(0, inplace=True)
                            x_test.fillna(0, inplace=True)
                            y_test.fillna(0, inplace=True)

                            reg = ElasticNet()
                            reg.fit(x_train, y_train)
                            y_pred = reg.predict(x_test)
                            y_pred = pd.DataFrame(y_pred)
                            y_pred = y_pred.apply(np.floor)
                            accuracy = accuracy_score(y_test, y_pred) * 100
                            precision = precision_score(y_test, y_pred, average='weighted') * 100
                            recall = recall_score(y_test, y_pred, average='weighted') * 100
                            f1 = f1_score(y_test, y_pred, average='weighted') * 100

                            table["Algorithm"].append("Elastic Net")
                            table["Precision"].append(precision)
                            table["Recall"].append(recall)
                            table["F1-Score"].append(f1)
                            pickle.dump(reg, open('EN.pkl', 'wb'))

                        elif algorithm == "Lasso Regression":
                            x_train = pd.DataFrame(x_train)
                            y_train = pd.DataFrame(y_train)
                            x_test = pd.DataFrame(x_test)
                            y_test = pd.DataFrame(y_test)

                            x_train = x_train.apply(pd.to_numeric, errors='coerce')
                            y_train = y_train.apply(pd.to_numeric, errors='coerce')
                            x_test = x_test.apply(pd.to_numeric, errors='coerce')
                            y_test = y_test.apply(pd.to_numeric, errors='coerce')
                            x_train.fillna(0, inplace=True)
                            y_train.fillna(0, inplace=True)
                            x_test.fillna(0, inplace=True)
                            y_test.fillna(0, inplace=True)

                            reg = Lasso()
                            reg.fit(x_train, y_train)
                            y_pred = reg.predict(x_test)
                            y_pred = pd.DataFrame(y_pred)
                            y_pred = y_pred.apply(np.floor)
                            accuracy = accuracy_score(y_test, y_pred) * 100
                            precision = precision_score(y_test, y_pred, average='weighted') * 100
                            recall = recall_score(y_test, y_pred, average='weighted') * 100
                            f1 = f1_score(y_test, y_pred, average='weighted') * 100

                            table["Algorithm"].append("Lasso Regression")
                            table["Precision"].append(precision)
                            table["Recall"].append(recall)
                            table["F1-Score"].append(f1)
                            pickle.dump(reg, open('LAR.pkl', 'wb'))
                    if st.button('Run Modelling'):
                        df_results = pd.DataFrame(table)

        with col2:
            st.subheader("Unsupervised machine learning")
            numerical_col = df.select_dtypes(include=np.number).columns
            categorical_col = df.select_dtypes(exclude=np.number).columns
            ct_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_col)],
                                           remainder='passthrough')
            df = ct_encoder.fit_transform(df)
            choice1 = st.selectbox("Unsupervised", ["Clustering", "Dimensionality Reduction"])
            if "Clustering" in choice1:
                algorithms = st.multiselect("Clustering Algorithms",
                                            ["Affinity Propagation", "Agglomerative Clustering",
                                             "BIRCH", "DBSCAN", "K-Means", "Mini-Batch K-Means",
                                             "Mean Shift", "OPTICS", "Spectral Clustering",
                                             "Gaussian Mixture Model"])

                if st.button('Run Model'):
                    snow = True
                    table = {"Algorithm": [], "Silhouette": []}
                    for algorithm in algorithms:
                        if algorithm == "Affinity Propagation":
                            clustering = AffinityPropagation()
                            clustering.fit(df)
                            labels = clustering.labels_
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('AP.pkl', 'wb'))

                        elif algorithm == "Agglomerative Clustering":
                            clustering = AgglomerativeClustering()
                            clustering.fit(df)
                            labels = clustering.labels_
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('AC.pkl', 'wb'))

                        elif algorithm == "BIRCH":
                            clustering = Birch()
                            clustering.fit(df)
                            labels = clustering.labels_
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('BC.pkl', 'wb'))

                        elif algorithm == "DBSCAN":
                            clustering = DBSCAN()
                            labels = clustering.fit_predict(df)
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('DB.pkl', 'wb'))

                        elif algorithm == "K-Means":
                            clustering = KMeans()
                            clustering.fit(df)
                            labels = clustering.predict(df)
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_avg = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append("K-Means")
                            table["Silhouette"].append(silhouette_avg)
                            pickle.dump(clustering, open('KM.pkl', 'wb'))

                        elif algorithm == "Mini-Batch K-Means":
                            clustering = MiniBatchKMeans()
                            clustering.fit(df)
                            labels = clustering.labels_
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('MBK.pkl', 'wb'))

                        elif algorithm == "Mean Shift":
                            clustering = MeanShift()
                            labels = clustering.fit_predict(df)
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('MS.pkl', 'wb'))

                        elif algorithm == "OPTICS":
                            clustering = OPTICS()
                            clustering.fit(df)
                            labels = clustering.labels_
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('OC.pkl', 'wb'))

                        elif algorithm == "Spectral Clustering":
                            clustering = SpectralClustering()
                            clustering.fit(df)
                            labels = clustering.labels_
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('SC.pkl', 'wb'))

                        elif algorithm == "Gaussian Mixture Model":
                            clustering = GaussianMixture(n_components=10)
                            labels = clustering.fit_predict(df)
                            uniq = len(np.unique(labels))
                            if uniq == 1:
                                st.warning("Improper Data for " + algorithm)
                                continue
                            silhouette_score = metrics.silhouette_score(df, labels) * 100
                            table["Algorithm"].append(algorithm)
                            table["Silhouette"].append(silhouette_score)
                            pickle.dump(clustering, open('GMM.pkl', 'wb'))
                    df_results = pd.DataFrame(table)

            elif "Dimensionality Reduction" in choice1:
                algorithms = st.selectbox("Dimensionality Reduction",
                                          ["PCA", "LDA", "Truncated SVD", "t-SNE", "MDS", "Isomap"])
                scaler = StandardScaler()
                X_train_std = scaler.fit_transform(x_train)
                X_test_std = scaler.transform(x_test)

                nc = st.slider("n_components", 1, df_clone.shape[1])
                if st.button('Run Model'):
                    snow = True
                    if algorithms == "PCA":
                        pca = PCA(n_components=nc)
                        data_pca = pca.fit_transform(df)
                        st.write("PCA Results:")
                        df_results = pd.DataFrame(data_pca)

                    elif algorithms == "LDA":
                        lda = LDA(n_components=2)
                        data_lda = lda.fit_transform(X_train_std, y_train)
                        st.write("LDA Results:")
                        df_results = pd.DataFrame(data_lda)
                    elif algorithms == "Truncated SVD":
                        svd = TruncatedSVD(n_components=nc)
                        data_svd = svd.fit_transform(df)
                        st.write("Truncated SVD Results:")
                        df_results = pd.DataFrame(data_svd)

                    elif algorithms == "t-SNE":
                        tsne = TSNE(n_components=nc, random_state=42)
                        data_tsne = tsne.fit_transform(df)
                        st.write("t-SNE Results:")
                        df_results = pd.DataFrame(data_tsne)

                    elif algorithms == "MDS":
                        mds = MDS(n_components=nc, random_state=42)
                        data_mds = mds.fit_transform(df)
                        st.write("MDS Results:")
                        df_results = pd.DataFrame(data_mds)

                    elif algorithms == "Isomap":
                        isomap = Isomap(n_components=nc, n_neighbors=5)
                        data_isomap = isomap.fit_transform(df)
                        st.write("Isomap Results:")
                        df_results = pd.DataFrame(data_isomap)

        st.write(df_results)
        if snow:
            st.snow()

if choice == "Download":
    if not os.path.exists('./data.csv'):
        st.subheader("Go To Upload File")
    else:
        df = df.iloc[:, :]
        # st.dataframe(df.head(10))
        df.to_csv(r"DATA.csv")
        with open('DATA.csv', 'rb') as file:
            data = file.read()
        st.download_button(
            label="Download Final Dataset",
            data=data,
            file_name='DATA.csv',
            mime='text/csv',
        )
    d = {"LOR": "Logistic Regression", "DT": "Decision Trees", "RF": "Random Forest", "NB": "Naive Bayes",
         "SVM": "Support Vector Machines (SVM)",
         "GB": "Gradient Boosting", "NN": "Neural Networks", "QDA": "Quadratic Discriminant Analysis (QDA)",
         "AB": "Adaptive Boosting (AdaBoost)", "GP": "Gaussian Processes", "PT": "Perceptron", "KNC": "KNN Classifier",
         "RC": "Ridge Classifier", "PA": "Passive Aggressive Classifier", "EN": "Elastic Net",
         "LAR": "Lasso Regression",
         "LR": "Linear Regression", "PR": "Polynomial Regression",
         "SVR": "Support Vector Regression", "DTR": "Decision Tree Regression",
         "RFR": "Random Forest Regression", "RR": "Ridge Regression",
         "LASR": "Lasso Regression", "GR": "Gaussian Regression", "KNR": "KNN Regression", "ABR": "AdaBoost",
         "AP": "Affinity Propagation", "AC": "Agglomerative Clustering",
         "BC": "BIRCH", "DB": "DBSCAN", "KM": "K-Means", "MBK": "Mini-Batch K-Means",
         "MS": "Mean Shift", "OC": "OPTICS", "SC": "Spectral Clustering",
         "GMM": "Gaussian Mixture Model"
         }
    cla_model = ['LOR', 'DT', 'RF', 'NB', 'SVM', 'GB', 'NN', 'QDA', 'AB', 'GP', 'PT', 'RC', 'PA', 'EN', 'LAR', 'KNC']
    reg_model = ['LR', 'PR', 'SVR', 'DTR', 'RFR', 'RR', 'LASR', 'GR', 'ABR', 'KNR']
    clu_model = ['AP', 'AC', 'BC', 'DB', 'KM', 'MBK', 'MS', 'OC', 'SC', 'GMM']
    choicel = st.selectbox("Model", ["Regression", "Classification", "Clustering"])

    if choicel == "Regression":
        available_models = []
        for i in reg_model:
            if os.path.exists("./" + i + ".pkl"):
                with open(i + '.pkl', 'rb') as file:
                    available_models.append([i, d[i]])
                    data = file.read()
                st.download_button(
                    label=d[i],
                    data=data,
                    file_name="./" + i + ".pkl"
                )
        if st.checkbox("Do Prediction"):
            if "chosen_target" in st.session_state:
                chosen_target = st.session_state['chosen_target']
                cols = df.columns
                predict = []
                minmax = {}
                for col in cols:
                    if col == chosen_target:
                        continue
                    if col in st.session_state['numerical_col_set']:
                        mv = st.session_state['minmaxtable'][col]
                        x = st.number_input(col, min_value=mv[0], max_value=mv[1])
                        v = (x - mv[0]) / (mv[1] - mv[0])
                        predict += [v]
                    elif col in st.session_state['categorical_col_set']:
                        uniquevals = df[col].unique()
                        x = st.selectbox(col, uniquevals)
                        v = []
                        for i in uniquevals:
                            if i == x:
                                v += [1]
                            else:
                                v += [0]
                        predict += v

                npredict = np.array(predict).reshape(1, -1)

                model = st.selectbox("Select the model to predict", available_models)
                st.write("Selected Model is ", model[1])
                with open(model[0] + '.pkl', 'rb') as f:
                    mod = pickle.load(f)
                predictions = mod.predict(npredict)[0]
                st.subheader("predicted Class " + str(predictions))

            else:
                st.error("Select the chosen target in Modelling page", icon="🚨")

    if choicel == "Classification":
        available_models = []
        for i in cla_model:
            if os.path.exists("./" + i + ".pkl"):
                with open(i + '.pkl', 'rb') as file:
                    available_models.append([i, d[i]])
                    data = file.read()
                st.download_button(
                    label=d[i],
                    data=data,
                    file_name="./" + i + ".pkl"
                )
        if st.checkbox("Do Classification"):
            if "chosen_target" in st.session_state:
                chosen_target = st.session_state['chosen_target']
                cols = df.columns
                predict = []
                minmax = {}
                for col in cols:
                    if col == chosen_target:
                        continue
                    if col in st.session_state['numerical_col_set']:
                        mv = st.session_state['minmaxtable'][col]
                        x = st.number_input(col, min_value=mv[0], max_value=mv[1])
                        v = (x - mv[0]) / (mv[1] - mv[0])
                        predict += [v]
                    elif col in st.session_state['categorical_col_set']:
                        uniquevals = df[col].unique()
                        x = st.selectbox(col, uniquevals)
                        v = []
                        for i in uniquevals:
                            if i == x:
                                v += [1]
                            else:
                                v += [0]
                        predict += v
                npredict = np.array(predict).reshape(1, -1)
                # st.write(predict)
                model = st.selectbox("Select the model to classify", available_models)
                st.write("Selected Model is ", model[1])
                with open(model[0] + '.pkl', 'rb') as f:
                    mod = pickle.load(f)
                predictions = int(mod.predict(npredict)[0])
                classes = st.session_state['classes']
                # st.write(predictions,classes)
                st.subheader("predicted Class " + classes[predictions])

            else:
                st.error("Select the chosen target in Modelling page", icon="🚨")

    if choicel == "Clustering":
        for i in clu_model:
            if os.path.exists("./" + i + ".pkl"):
                with open(i + '.pkl', 'rb') as file:
                    data = file.read()
                st.download_button(
                    label=d[i],
                    data=data,
                    file_name="./" + i + ".pkl"
                )
