# ==============================
# Import libraries
#===============================
import os
print('current working directory:', os.getcwd())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.metrics import accuracy_score, recall_score, auc,precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_curve
pd.set_option('display.max_columns', None)


# ==============================
# Data cleaning
# #===============================

## Load datasets
#train
df1 = pd.read_csv('C:/Users/HP/Documents/YDP/AI Bias Hackathon/loan_access_dataset.csv')
df = df1.copy()
df.head()

#test
test = pd.read_csv('C:/Users/HP/Documents/YDP/AI Bias Hackathon/test.csv')


## check missing values
print(df.isnull().sum())
print(test.isnull().sum())


## Identify and check for duplicates
duplicates = df[df.duplicated()]


## Map target feature (Loan_Approved) to 0 and 1
df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})

# ==============================
# Data Preprocessing and Feature Engineering
# ===============================

## check dataset information and data types.
df.info()

## check summary statistics
df.describe()

## check values count and proportions for categorical features
cat_cols = ['Gender', 'Race', 'Citizenship_Status', 'Disability_Status', 'Criminal_Record', 'Age_Group', 'Zip_Code_Group', 'Language_Proficiency']

for col in cat_cols:
    print(f'\n --- {col} (Counts and Proportions) ---')
    print(df[col].value_counts(normalize=False))
    print(df[col].value_counts(normalize=True).round(2))
          


#plot a pie chart to show target value distribution 
class_counts = df['Loan_Approved'].value_counts()

sizes = class_counts.values
labels = ['Denied' if val == 0 else 'Approved' for val in class_counts.index]

plt.figure(figsize=(5, 5))
plt.pie(sizes,labels=labels, autopct='%.1f%%', startangle=45, colors= ['skyblue', 'lightcoral'])

plt.title('Distribution of Loan Approval')
plt.axis('equal')
plt.show()


## Create a temporary dataset to visualize correlation
temp_df = df1.copy()

## Find the categorical columns
categorical_columns = temp_df.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in categorical_columns:
    temp_df[col] = label_encoder.fit_transform(temp_df[col].astype(str))

temp_df =temp_df.drop('ID', axis=1)

##visualize correlation

plt.figure(figsize=(10,8))
sns.heatmap(temp_df.corr(method='spearman'), annot=True, fmt='.2f', cmap='coolwarm', annot_kws={'fontsize': 10}, vmin=-1, vmax=1, center=0)
plt.title('Loan Approval Heatmap')
#plt.savefig('heatmap of Loan features.png', dpi=300, bbox_inches='tight')
plt.show()



## ========================================================
## Bias Detection: Inspection of Bias on an Individual Basis
## =========================================================
# Some cases are suspicious so they will be flagged e.g high credit score, loan denied
## define limits 

high_credit = df['Credit_Score'].quantile(0.85)
low_credit = df['Credit_Score'].quantile(0.15)
high_income = df['Income'].quantile(0.85)
low_income = df['Income'].quantile(0.15)

## function to flag those cases
def abnormal_cases(df, high_credit, low_credit, high_income, low_income):
    """
    this function flags suspicious cases where:
    1. High credit score but denied
    2. Low credit score but approved
    3. High income but denied
    4. Low income but approved.
    """
    # Create boolean flags for each abnormal condition
    high_credit_denied = (df['Credit_Score'] > high_credit) & (df['Loan_Approved'] == 0)
    low_credit_approved = (df['Credit_Score'] < low_credit) & (df['Loan_Approved'] == 1)
    high_income_denied = (df['Income'] > high_income) & (df['Loan_Approved'] == 0)
    low_income_approved = (df['Income'] < low_income) & (df['Loan_Approved'] == 1)


    df['Suspicious'] = high_credit_denied | low_credit_approved | high_income_denied | low_income_approved
    # Return only suspicious rows for review
    return df[df['Suspicious'] == True].sort_values(by='Credit_Score', ascending=False)

## call the function
abnormal = abnormal_cases(df, high_credit, low_credit, high_income, low_income)
abnormal

## Drop irrelevant features and new features that were created
df = df.drop(columns=['ID', 'Age_Group', 'CreditScore_Group', 'Suspicious'])

 
## =====================
# Model Development
# I will be training two models. One where I exclude the sensitive features to remoe any 
# form of bias then another one where I include sensitive features for bias auditing.
# sensitive features (Race, Gender, Disability_Status etc). I will 
# then compare both.
## ====================

## Random Forest will be used to develop the models
# Define the model.
rf_s = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

rf_n = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

## define sensitive and non-sensitive features ('s' represents 
#sensitive, 'ns' represents non-sensitive)
# Group features into sensitive and predictive( what is exactly needed for a loan prediction)

sensitive_features = ['Gender', 'Race', 'Citizenship_Status','Language_Proficiency', 
                      'Criminal_Record', 'Zip_Code_Group'] 
predictive_features = ['Age','Income', 'Credit_Score', 'Loan_Amount',
                      'Employment_Type', 'Education_Level']



## =======================
# Model with sensitive features:The steps below are for the model that contains sensitive attributes.
#=========================
# define features for training 
num_cols_s = ['Age', 'Income','Credit_Score','Loan_Amount']
cat_cols_s = ['Gender','Race','Employment_Type','Education_Level',
                   'Citizenship_Status','Language_Proficiency','Disability_Status',
                    'Criminal_Record','Zip_Code_Group' ]

#define X and y train
X_s= df[num_cols_s + cat_cols_s]
y_target= df['Loan_Approved']

#split dataframe with stratified sampling
train_idx, val_idx = train_test_split(
    df.index, test_size=0.2, stratify=df['Loan_Approved'], random_state=42
)

X_train_s = df.loc[train_idx, num_cols_s + cat_cols_s]
X_val_s = df.loc[val_idx,num_cols_s + cat_cols_s]
y_train_s = df.loc[train_idx, 'Loan_Approved']
y_val_s = df.loc[val_idx, 'Loan_Approved']


# Categorical columns are to be encoded
categorical_transformer_s = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
#numerical columns are not being scaled because random forest is a tree based model and can handle the values


preprocessor_s = ColumnTransformer([
    ('num', 'passthrough', num_cols_s),
    ('cat', categorical_transformer_s, cat_cols_s)
])

# random forest pipeline for model with sensitive features
rf_pipeline_s = Pipeline([
    ('preprocessor_s', preprocessor_s),
    ('model_s', rf_s)
])

#train and evaluate random forest on model with sensistive features
rf_pipeline_s.fit(X_train_s, y_train_s)
rf_preds_s = rf_pipeline_s.predict(X_val_s)
rf_probs_s = rf_pipeline_s.predict_proba(X_val_s)[:, 1]

print("--- Random Forest Results For Model with Sensitive Features ---")
print(classification_report(y_val_s, rf_preds_s))
print("AUC:", roc_auc_score(y_val_s, rf_probs_s))


## =======================
# Model without sensitive features:The steps below are for the model that do not contain sensitive attributes.
# The features used for modeling are: Age,Income, 'redit_Score, Loan_Amount, Employment_Type, Education_Level
#validation set will be used to see the model's performance
#=========================
# define features for training 

num_cols_n = ['Age', 'Income','Credit_Score','Loan_Amount']
cat_cols_n = ['Employment_Type','Education_Level']

# define x and y
X_n = df[num_cols_n + cat_cols_n]
y_target = df['Loan_Approved']

#data has been split above. Reusing indices
X_train_n = df.loc[train_idx, num_cols_n + cat_cols_n]
X_val_n = df.loc[val_idx,num_cols_n + cat_cols_n]
y_train_n = df.loc[train_idx, 'Loan_Approved']
y_val_n = df.loc[val_idx, 'Loan_Approved']

#categorical columns to be encoded
categorical_transformer_n = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_n = ColumnTransformer([
    ('num', 'passthrough', num_cols_n),
    ('cat', categorical_transformer_n, cat_cols_n)
    ])

#define X test
X_test = test[num_cols_n + cat_cols_n]

rf_pipeline_n = Pipeline([
    ('preprocess_n',preprocessor_n ),
    ('model_n', rf_n)
])

#Train and evaluate Random Forest on model without sensitive features
rf_pipeline_n.fit(X_train_n, y_train_n)
rf_preds_n = rf_pipeline_n.predict(X_val_n)
rf_probs_n = rf_pipeline_n.predict_proba(X_val_n)[:, 1]

print('--- Random Forest Results for Model without Sensitive Features ---')
print(classification_report(y_val_n, rf_preds_n))
print("AUC:", roc_auc_score(y_val_n, rf_probs_n))

#Re-train random forest on full train dataset
X_full_train = df[num_cols_n + cat_cols_n]
y_full_train = df['Loan_Approved']

rf_pipeline_n.fit(X_full_train, y_full_train)

test_preds_n = rf_pipeline_n.predict(X_test)

#add prediction values to submission file
submission = pd.DataFrame({
    'ID': test['ID'],
    'Loan_Approved': test_preds_n
})

submission.to_csv('submission.csv', index=False)


## SHAP analysis
# Two plots will be plotted (one for the sensitive and non sensitive models each)



#for sensitive
# Extract the preprocessor and model from the pipeline
preprocessor_s = rf_pipeline_s.named_steps['preprocessor_s']
model_s = rf_pipeline_s.named_steps['model_s']

# Transform X_val_s for SHAP analysis
X_val_s_transformed = preprocessor_s.transform(X_val_s)

# Get transformed feature names
feature_names_s = preprocessor_s.get_feature_names_out()

# Create a DataFrame for SHAP with feature names (keep full names for consistency with pipeline)
X_val_s_df = pd.DataFrame(X_val_s_transformed.toarray() if hasattr(X_val_s_transformed, 'toarray') else X_val_s_transformed,
                          columns=feature_names_s,
                          index=X_val_s.index)

# Train a SHAP explainer and compute values
explainer_s = shap.TreeExplainer(model_s)
shap_values_s = explainer_s.shap_values(X_val_s_df)
shap_class_1_s = shap_values_s[:, :, 1]

# CLEAN feature names for plotting only
clean_names_s = [name.replace('cat__', '').replace('num__', '').replace('remainder__', '') for name in feature_names_s]


# Plot SHAP summary
shap.summary_plot(shap_class_1_s, X_val_s_df, feature_names=clean_names_s, show=False)

plt.title('SHAP Summary: Sensitive Model')
plt.tight_layout()
plt.show()


#for non sensistive
# Extract the preprocessor and model from the pipeline
preprocessor_n = rf_pipeline_n.named_steps['preprocess_n']
model_n = rf_pipeline_n.named_steps['model_n']

# Transform X_val_n for SHAP analysis
X_val_n_transformed = preprocessor_n.transform(X_val_n)

# Get transformed feature names
feature_names_n = preprocessor_n.get_feature_names_out()

# Create a DataFrame for SHAP with feature names (keep full names for group prediction later)
X_val_n_df = pd.DataFrame(X_val_n_transformed.toarray() if hasattr(X_val_n_transformed, 'toarray') else X_val_n_transformed,
                          columns=feature_names_n,
                          index=X_val_n.index)

# Train a SHAP explainer and compute values
explainer_n = shap.TreeExplainer(model_n)
shap_values_n = explainer_n.shap_values(X_val_n_df)
shap_class_1_n = shap_values_n[:, :, 1]

# CLEAN feature names for plotting only
clean_names_n = [name.replace('cat__', '').replace('num__', '').replace('remainder__', '') for name in feature_names_n]

# Plot SHAP summary
shap.summary_plot(shap_class_1_n, X_val_n_df, feature_names=clean_names_n, show=False)

plt.title('SHAP Summary: Non-sensitive Model')
plt.tight_layout()
plt.show()






# get sensitive features for the validation set
val_sensitive = df.loc[X_val_n.index, ['Gender', 'Race', 'Disability_Status', 'Criminal_Record']]

# make predictions on the preprocessed validation set
val_preds = rf_pipeline_n.predict(X_val_n)

# STEP 3: Combine sensitive attributes with predictions
val_sensitive_preds = val_sensitive.copy()
val_sensitive_preds['Predicted_Loan_Approval'] = val_preds

# STEP 4: Group-wise approval rates
# Gender
gender_approval = val_sensitive_preds.groupby('Gender')['Predicted_Loan_Approval'].mean().reset_index()
print('\nApproval Rates by Gender:\n', gender_approval)

# Race
race_approval = val_sensitive_preds.groupby('Race')['Predicted_Loan_Approval'].mean().reset_index()
print("\nApproval Rates by Race:\n", race_approval)

# Disability Status
disability_approval = val_sensitive_preds.groupby('Disability_Status')['Predicted_Loan_Approval'].mean().reset_index()
print("\nApproval Rates by Disability Status:\n", disability_approval)


# Criminal Record
Criminal_record_approval = val_sensitive_preds.groupby('Criminal_Record')['Predicted_Loan_Approval'].mean().reset_index()
print("\nApproval Rates by Criminal Record:\n", Criminal_record_approval)



# Dictionary to map group names to your approval rate DataFrames
group_data = {
    'Gender': gender_approval,
    'Race': race_approval,
    'Disability_Status': disability_approval,
    'Criminal_Record': Criminal_record_approval
}

# Loop through each sensitive attribute and plot
for group, data in group_data.items():
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=group, y='Predicted_Loan_Approval', data=data, palette='pastel')
    plt.title(f'Loan Approval Rate by {group}')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)

    # Add percentage labels on each bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, height + 0.02, f'{height:.1%}', ha='center')

    plt.tight_layout()
    plt.show()




# Function to calculate FPR and FNR for each group
def compute_error_rates(df, group_col, y_true, y_pred):
    """this function calculates the false
    positive rate and false negative rate for
    each group"""
    results = []
    for group in df[group_col].unique():
        idx = df[group_col] == group
        y_true_group = y_true[idx]
        y_pred_group = y_pred[idx]

        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

        results.append({'Group': group, 'FPR': round(fpr, 3), 'FNR': round(fnr, 3)})
    
    return pd.DataFrame(results)

# Apply to each sensitive groups
val_sensitive_preds['True_Label'] = y_val_n  # y_val_n is your true labels
groups = ['Gender', 'Race', 'Disability_Status', 'Criminal_Record']

for group_col in groups:
    print(f"\nError rates for {group_col}:")
    error_df = compute_error_rates(val_sensitive_preds, group_col,
                                   val_sensitive_preds['True_Label'],
                                   val_sensitive_preds['Predicted_Loan_Approval'])
    print(error_df)




# List of groups and their titles
groups = ['Gender', 'Race']
error_types = ['FPR', 'FNR']

# Loop over group types and error types
for group_col in groups:
    # Get error rates for this group
    error_df = compute_error_rates(val_sensitive_preds, group_col,
                                   val_sensitive_preds['True_Label'],
                                   val_sensitive_preds['Predicted_Loan_Approval'])
    
    # For each error type (FPR and FNR), plot a bar chart
    for error_type in error_types:
        plt.figure(figsize=(7, 4))
        ax = sns.barplot(x='Group', y=error_type, data=error_df, palette='pastel')

        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, height + 0.02, f'{height:.2f}', ha='center', fontsize=9)

        plt.title(f'{error_type} by {group_col}')
        plt.ylabel('Error Rate')
        plt.ylim(0, max(error_df[error_type]) + 0.1)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()





print("\n==== SCRIPT COMPLETED SUCCESSFULLY ====\n")
