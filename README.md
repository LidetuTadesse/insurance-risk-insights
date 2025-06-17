# Insurance Risk Insights

A data science project analyzing auto insurance data to uncover patterns in risk, profitability, and claims using statistical analysis, EDA, and modeling. Built with clean code, Git versioning, and CI/CD practices.

---

## Project Scope

- Loss ratio analysis across regions, demographics, and vehicle types
- Temporal trends in claims and premiums
- Feature engineering and statistical testing
- Predictive modeling and risk profiling
- Automation via GitHub Actions

---

## Setup

```bash
git clone https://github.com/<your-username>/insurance-risk-insights.git
cd insurance-risk-insights
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate (Windows)
pip install -r requirements.txt
🗂️ Structure
bash
Copy
Edit
├── notebooks/       # EDA and prototyping
├── src/             # Core logic and processing
├── scripts/         # Utilities and helpers
├── tests/           # Unit tests
├── .github/         # CI/CD workflows
├── .vscode/         # Dev environment config

Tech Stack
Python • Pandas • Seaborn • Scikit-learn • Git • GitHub Actions



## 📊 Business Objective

The goal is to analyze historical car insurance claims to:
- Optimize marketing strategy
- Identify low-risk segments for reduced premiums
- Recommend data-driven improvements to insurance offerings

## Key Areas of Analysis

### A/B Hypothesis Testing
- Risk differences across provinces, zip codes, and gender
- Profit margin differences across geographic regions

### Statistical Modeling & Machine Learning
- Linear regression per zipcode to predict total claims
- ML models to predict optimal premium values based on:
  - Car features
  - Owner demographics
  - Geographic information
  - Other relevant features

### Insurance Domain Knowledge
- Incorporated research on key insurance terminologies

## Methodologies
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Hypothesis testing (t-tests, ANOVA)
- Linear regression and predictive modeling
- Feature importance analysis and model interpretation

## Outcome
The final report includes detailed analysis, modeling results, and business recommendations aimed at helping AlphaCare Insurance Solutions tailor their insurance products to attract and retain low-risk customers.

## Skills Practiced
- Data Engineering (DE)
- Predictive Analytics (PA)
- Machine Learning Engineering (MLE)