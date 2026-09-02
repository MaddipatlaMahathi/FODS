# ================================================================
# DSA0402 – FUNDAMENTALS OF DATA SCIENCE
# BANK CUSTOMER SUBSCRIPTION PREDICTION AND CUSTOMER SEGMENTATION
# ================================================================

# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)

from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


# ================================================================
# 2. LOAD DATASET
# ================================================================

# Change this path if your CSV is stored somewhere else
FILE_PATH = "bank-full.csv"

df = pd.read_csv(FILE_PATH, sep=";")

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ================================================================
# 3. BASIC DATASET INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ================================================================
# 4. CHECK MISSING VALUES
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

print("\nTotal Missing Values:", df.isnull().sum().sum())


# ================================================================
# 5. CHECK DUPLICATES
# ================================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of Duplicate Records:", duplicate_count)


# ================================================================
# 6. CHECK UNIQUE VALUES
# ================================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique()[:20])


# ================================================================
# 7. TARGET VARIABLE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE ANALYSIS")
print("=" * 70)

target_counts = df["y"].value_counts()

print("\nSubscription Counts:")
print(target_counts)

subscription_percentage = df["y"].value_counts(normalize=True) * 100

print("\nSubscription Percentage:")
print(subscription_percentage)


# ================================================================
# 8. TARGET VARIABLE VISUALIZATION
# ================================================================

plt.figure(figsize=(7, 5))

plt.bar(
    target_counts.index,
    target_counts.values
)

plt.title("Bank Term Deposit Subscription")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")

for i, value in enumerate(target_counts.values):
    plt.text(i, value + 500, str(value), ha="center")

plt.tight_layout()
plt.show()


# ================================================================
# 9. SEPARATE NUMERICAL AND CATEGORICAL FEATURES
# ================================================================

numerical_features = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = df.select_dtypes(
    include=["object"]
).columns.tolist()

# Remove target from categorical features
if "y" in categorical_features:
    categorical_features.remove("y")

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ================================================================
# 10. DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

desc_stats = df[numerical_features].describe().T

print(desc_stats)


# ================================================================
# 11. MEAN, VARIANCE AND STANDARD DEVIATION
# ================================================================

print("\n" + "=" * 70)
print("MEAN, VARIANCE AND STANDARD DEVIATION")
print("=" * 70)

for column in numerical_features:

    mean_value = df[column].mean()
    variance_value = df[column].var()
    std_value = df[column].std()

    print(f"\n{column}")
    print(f"Mean       : {mean_value:.2f}")
    print(f"Variance   : {variance_value:.2f}")
    print(f"Std Dev    : {std_value:.2f}")


# ================================================================
# 12. COVARIANCE MATRIX
# ================================================================

print("\n" + "=" * 70)
print("COVARIANCE MATRIX")
print("=" * 70)

covariance_matrix = df[numerical_features].cov()

print(covariance_matrix)


# ================================================================
# 13. CORRELATION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

correlation_matrix = df[numerical_features].corr()

print(correlation_matrix.round(3))


# ================================================================
# 14. CORRELATION HEATMAP
# ================================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix of Numerical Features")

plt.tight_layout()
plt.show()


# ================================================================
# 15. NUMERICAL FEATURE DISTRIBUTIONS
# ================================================================

for column in numerical_features:

    plt.figure(figsize=(7, 5))

    plt.hist(
        df[column],
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ================================================================
# 16. CATEGORICAL FEATURE ANALYSIS
# ================================================================

selected_categorical = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "poutcome"
]

for column in selected_categorical:

    counts = df[column].value_counts()

    plt.figure(figsize=(9, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.title(f"Customer Distribution by {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Customers")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ================================================================
# 17. OUTLIER DETECTION USING IQR
# ================================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS USING IQR")
print("=" * 70)

outlier_summary = []

for column in numerical_features:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_summary.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers)
    })

outlier_df = pd.DataFrame(outlier_summary)

print(outlier_df)


# ================================================================
# 18. BOXPLOTS
# ================================================================

boxplot_features = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

for column in boxplot_features:

    plt.figure(figsize=(7, 5))

    plt.boxplot(df[column])

    plt.title(f"Boxplot of {column}")
    plt.ylabel(column)

    plt.tight_layout()
    plt.show()


# ================================================================
# 19. SUBSCRIBER VS NON-SUBSCRIBER ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("SUBSCRIBER VS NON-SUBSCRIBER ANALYSIS")
print("=" * 70)

grouped_statistics = df.groupby("y")[numerical_features].mean()

print("\nMean Values by Subscription Status:")
print(grouped_statistics.round(2))


# ================================================================
# 20. SUBSCRIBERS BY JOB
# ================================================================

job_subscription = pd.crosstab(
    df["job"],
    df["y"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("SUBSCRIPTION RATE BY JOB")
print("=" * 70)

print(job_subscription.round(2))


plt.figure(figsize=(10, 6))

job_yes_rate = job_subscription["yes"].sort_values()

plt.barh(
    job_yes_rate.index,
    job_yes_rate.values
)

plt.title("Term Deposit Subscription Rate by Job")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Job")

plt.tight_layout()
plt.show()


# ================================================================
# 21. SUBSCRIPTION BY EDUCATION
# ================================================================

education_subscription = pd.crosstab(
    df["education"],
    df["y"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("SUBSCRIPTION RATE BY EDUCATION")
print("=" * 70)

print(education_subscription.round(2))


# ================================================================
# 22. BALANCE VS SUBSCRIPTION
# ================================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="balance",
    by="y"
)

plt.title("Balance Distribution by Subscription Status")
plt.suptitle("")
plt.xlabel("Subscription")
plt.ylabel("Balance")

plt.tight_layout()
plt.show()


# ================================================================
# 23. DURATION VS SUBSCRIPTION
# ================================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="duration",
    by="y"
)

plt.title("Call Duration by Subscription Status")
plt.suptitle("")
plt.xlabel("Subscription")
plt.ylabel("Duration (seconds)")

plt.tight_layout()
plt.show()


# ================================================================
# 24. POINT ESTIMATE AND CONFIDENCE INTERVAL
# ================================================================

print("\n" + "=" * 70)
print("POINT ESTIMATE AND 95% CONFIDENCE INTERVAL")
print("=" * 70)

n = len(df)

successes = (df["y"] == "yes").sum()

p_hat = successes / n

confidence_level = 0.95

z_value = stats.norm.ppf(0.975)

standard_error = np.sqrt(
    (p_hat * (1 - p_hat)) / n
)

margin_error = z_value * standard_error

lower_ci = p_hat - margin_error
upper_ci = p_hat + margin_error

print(f"Total Customers       : {n}")
print(f"Subscribed Customers   : {successes}")
print(f"Point Estimate        : {p_hat:.4f}")
print(f"Subscription Rate     : {p_hat * 100:.2f}%")
print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")
print(
    f"95% CI in Percentage  : "
    f"({lower_ci * 100:.2f}%, {upper_ci * 100:.2f}%)"
)


# ================================================================
# 25. PREPARE DATA FOR MACHINE LEARNING
# ================================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING PREPARATION")
print("=" * 70)

X = df.drop("y", axis=1)

y = df["y"].map({
    "no": 0,
    "yes": 1
})

print("\nTarget Encoding:")
print("no  -> 0")
print("yes -> 1")

print("\nTarget Distribution:")
print(y.value_counts())


# ================================================================
# 26. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])


# ================================================================
# 27. PREPROCESSING PIPELINE
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            RobustScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ================================================================
# 28. MODEL 1 – K NEAREST NEIGHBORS
# ================================================================

knn_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]
)


# ================================================================
# 29. MODEL 2 – DECISION TREE
# ================================================================

decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=8,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ================================================================
# 30. MODEL 3 – LOGISTIC REGRESSION
# ================================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ================================================================
# 31. TRAIN MODELS
# ================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

models = {
    "KNN": knn_model,
    "Decision Tree": decision_tree_model,
    "Logistic Regression": logistic_model
}

results = []

predictions = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    predictions[model_name] = y_pred

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"{model_name} training completed.")


# ================================================================
# 32. MODEL PERFORMANCE COMPARISON
# ================================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# ================================================================
# 33. PERFORMANCE VISUALIZATION
# ================================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = np.arange(len(results_df["Model"]))

width = 0.18

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):

    plt.bar(
        x + i * width,
        results_df[metric],
        width,
        label=metric
    )

plt.xticks(
    x + width * 1.5,
    results_df["Model"]
)

plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.title("Machine Learning Model Performance Comparison")

plt.legend()

plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ================================================================
# 34. FIND BEST MODEL
# ================================================================

best_model_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Model Based on F1 Score:",
    best_model_row["Model"]
)

print(
    "Accuracy :",
    round(best_model_row["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_model_row["Precision"], 4)
)

print(
    "Recall   :",
    round(best_model_row["Recall"], 4)
)

print(
    "F1 Score :",
    round(best_model_row["F1 Score"], 4)
)


# ================================================================
# 35. CLASSIFICATION REPORTS
# ================================================================

for model_name in models.keys():

    print("\n" + "=" * 70)
    print(f"CLASSIFICATION REPORT – {model_name}")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions[model_name],
            target_names=[
                "Not Subscribed",
                "Subscribed"
            ],
            zero_division=0
        )
    )


# ================================================================
# 36. CONFUSION MATRICES
# ================================================================

for model_name in models.keys():

    cm = confusion_matrix(
        y_test,
        predictions[model_name]
    )

    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX – {model_name}")
    print("=" * 70)

    print(cm)

    plt.figure(figsize=(6, 5))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"Confusion Matrix – {model_name}"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Not Subscribed", "Subscribed"]
    )

    plt.yticks(
        [0, 1],
        ["Not Subscribed", "Subscribed"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


# ================================================================
# 37. K-MEANS CLUSTERING
# ================================================================

print("\n" + "=" * 70)
print("K-MEANS CUSTOMER SEGMENTATION")
print("=" * 70)

# Selected customer behaviour/demographic features
cluster_features = [
    "age",
    "balance",
    "duration",
    "campaign",
    "previous"
]

cluster_data = df[cluster_features].copy()

# Scaling
cluster_scaler = RobustScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)


# ================================================================
# 38. ELBOW METHOD
# ================================================================

k_values = range(2, 9)

inertia_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(cluster_scaled)

    inertia_values.append(
        kmeans.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Selecting K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")

plt.xticks(list(k_values))

plt.tight_layout()
plt.show()


# ================================================================
# 39. SILHOUETTE ANALYSIS
# ================================================================

silhouette_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        cluster_scaled
    )

    score = silhouette_score(
        cluster_scaled,
        labels
    )

    silhouette_values.append(score)

print("\n" + "=" * 70)
print("SILHOUETTE SCORES")
print("=" * 70)

for k, score in zip(
    k_values,
    silhouette_values
):

    print(
        f"K = {k}  -->  "
        f"Silhouette Score = {score:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(list(k_values))

plt.tight_layout()
plt.show()


# ================================================================
# 40. SELECT BEST K USING SILHOUETTE SCORE
# ================================================================

best_k = list(k_values)[
    np.argmax(silhouette_values)
]

print("\nBest K Based on Silhouette Score:", best_k)


# ================================================================
# 41. FINAL K-MEANS MODEL
# ================================================================

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(
    cluster_scaled
)


# ================================================================
# 42. CLUSTER DISTRIBUTION
# ================================================================

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\n" + "=" * 70)
print("CUSTOMER CLUSTER DISTRIBUTION")
print("=" * 70)

print(cluster_counts)


plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ================================================================
# 43. CLUSTER PROFILE
# ================================================================

print("\n" + "=" * 70)
print("CLUSTER PROFILE")
print("=" * 70)

cluster_profile = df.groupby(
    "Cluster"
)[cluster_features].mean()

print(
    cluster_profile.round(2)
)


# ================================================================
# 44. SUBSCRIPTION RATE BY CLUSTER
# ================================================================

cluster_subscription = pd.crosstab(
    df["Cluster"],
    df["y"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("SUBSCRIPTION RATE BY CUSTOMER CLUSTER")
print("=" * 70)

print(
    cluster_subscription.round(2)
)


# ================================================================
# 45. CLUSTER VISUALIZATION – AGE VS BALANCE
# ================================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["age"],
    df["balance"],
    c=df["Cluster"],
    alpha=0.5
)

plt.title(
    "Customer Segmentation: Age vs Balance"
)

plt.xlabel("Age")
plt.ylabel("Balance")

plt.colorbar(
    label="Cluster"
)

plt.tight_layout()
plt.show()


# ================================================================
# 46. CLUSTER VISUALIZATION – DURATION VS BALANCE
# ================================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["duration"],
    df["balance"],
    c=df["Cluster"],
    alpha=0.5
)

plt.title(
    "Customer Segmentation: Call Duration vs Balance"
)

plt.xlabel("Call Duration")
plt.ylabel("Balance")

plt.colorbar(
    label="Cluster"
)

plt.tight_layout()
plt.show()


# ================================================================
# 47. IDENTIFY HIGH-POTENTIAL CLUSTER
# ================================================================

cluster_analysis = cluster_profile.copy()

cluster_analysis["Subscription_Rate"] = (
    cluster_subscription["yes"]
)

print("\n" + "=" * 70)
print("COMPLETE CLUSTER ANALYSIS")
print("=" * 70)

print(
    cluster_analysis.round(2)
)


high_potential_cluster = (
    cluster_analysis["Subscription_Rate"].idxmax()
)

low_response_cluster = (
    cluster_analysis["Subscription_Rate"].idxmin()
)

print(
    "\nHigh-Potential Cluster:",
    high_potential_cluster
)

print(
    "Low-Response Cluster:",
    low_response_cluster
)


# ================================================================
# 48. SORT CUSTOMERS BY BALANCE
# ================================================================

print("\n" + "=" * 70)
print("TOP CUSTOMERS BY BALANCE")
print("=" * 70)

top_balance_customers = df.sort_values(
    by="balance",
    ascending=False
).head(10)

print(
    top_balance_customers[
        [
            "age",
            "job",
            "balance",
            "housing",
            "loan",
            "duration",
            "y",
            "Cluster"
        ]
    ]
)


# ================================================================
# 49. GROUPING AND AGGREGATION
# ================================================================

print("\n" + "=" * 70)
print("JOB-WISE CUSTOMER SUMMARY")
print("=" * 70)

job_summary = df.groupby("job").agg(
    Customer_Count=("age", "count"),
    Average_Age=("age", "mean"),
    Average_Balance=("balance", "mean"),
    Average_Duration=("duration", "mean")
)

print(
    job_summary.round(2)
)


# ================================================================
# 50. FINAL CUSTOMER SEGMENT SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL CUSTOMER SEGMENT SUMMARY")
print("=" * 70)

for cluster in sorted(df["Cluster"].unique()):

    cluster_data_df = df[
        df["Cluster"] == cluster
    ]

    avg_age = cluster_data_df["age"].mean()
    avg_balance = cluster_data_df["balance"].mean()
    avg_duration = cluster_data_df["duration"].mean()
    avg_campaign = cluster_data_df["campaign"].mean()

    subscription_rate = (
        cluster_data_df["y"]
        .eq("yes")
        .mean() * 100
    )

    print(f"\nCluster {cluster}")
    print(f"Customers             : {len(cluster_data_df)}")
    print(f"Average Age           : {avg_age:.2f}")
    print(f"Average Balance       : {avg_balance:.2f}")
    print(f"Average Call Duration : {avg_duration:.2f}")
    print(f"Average Campaign      : {avg_campaign:.2f}")
    print(f"Subscription Rate     : {subscription_rate:.2f}%")


# ================================================================
# 51. SAVE RESULTS TO CSV
# ================================================================

results_df.to_csv(
    "model_performance.csv",
    index=False
)

outlier_df.to_csv(
    "outlier_analysis.csv",
    index=False
)

cluster_profile.to_csv(
    "cluster_profile.csv"
)

cluster_subscription.to_csv(
    "cluster_subscription_rates.csv"
)

job_summary.to_csv(
    "job_summary.csv"
)

print("\n" + "=" * 70)
print("RESULT FILES SAVED")
print("=" * 70)

print("model_performance.csv")
print("outlier_analysis.csv")
print("cluster_profile.csv")
print("cluster_subscription_rates.csv")
print("job_summary.csv")


# ================================================================
# 52. FINAL CONCLUSION
# ================================================================

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

print("""
1. The Bank Marketing dataset contains customer demographic,
   financial and campaign-related information.

2. The target variable 'y' indicates whether the customer
   subscribed to the term deposit.

3. The dataset is imbalanced because non-subscribers are much
   more numerous than subscribers.

4. Missing values and duplicate records were checked.

5. Numerical and categorical variables were analysed using
   descriptive statistics and visualizations.

6. Mean, variance, covariance and correlation were calculated
   for numerical variables.

7. A 95% confidence interval was calculated for the overall
   subscription proportion.

8. Three classification algorithms were implemented:
   K-Nearest Neighbors, Decision Tree and Logistic Regression.

9. Accuracy, precision, recall and F1-score were used to
   compare the classification models.

10. K-Means clustering was used to divide customers into
    meaningful customer segments.

11. The Elbow Method and Silhouette Score were used to select
    an appropriate number of clusters.

12. Customer clusters were compared with subscription rates
    to identify high-potential and low-response segments.

13. The analysis can help a bank target customers more
    effectively and improve marketing campaign efficiency.
""")

print("\nPROGRAM COMPLETED SUCCESSFULLY!")