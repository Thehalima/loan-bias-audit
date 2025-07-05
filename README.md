# Bias Detection and Audit in AI Loan Approvals
The task is to analyse a loan access dataset, develop AI models for bias, assess fairness and recommend suggestions to improve ethical and responsible AI development. 
## What the Project Does
* Trains Random Forest models to predict loan approval.
* Compares results with and without sensitive features.
* Performs bias detection using:
    * Approval rate comparisons by group.
    * SHAP explainability.
    * False Positive and Negative Rate analysis.
 
## Project Structure
```
loan-bias-audit/
├── Loan.py                # Main Python script
├── submission.csv         # Model predictions output
├── charts/                # Plots from the analysis
├── ai_risk_report.docx    # Full bias audit report
└── README.md              # This file
```
 
## Video Demonstration
Watch the video demo here  [youtube link] (https://youtu.be/Oj-Xq5Ho5Do)

## Key Findings
* Approval rate disparities across Race and Gender.
* False Negative rates show more wrongful rejections for Multiracial and Native American groups.
* Removing sensitive features alone did not fully remove bias.

## Limitations
* Bias mitigation techniques could not be applied due to insufficient time.
* Bias was found even after excluding sensitive features. It shows that proxy variables still carry bias.
