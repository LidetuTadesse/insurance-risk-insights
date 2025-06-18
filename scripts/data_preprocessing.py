# importing libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def load_and_prepare_data(path):
    """
    Loads the insurance dataset and prepares it for modeling by:
        - Removing irrelevant or high-missing columns
        - Handling missing values via median imputation
        - One-hot encoding categorical variables
        - Filtering claim-only data for severity prediction

    Returns:
        - df: Preprocessed full dataset
        - claim_data: Subset containing only records with claims (TotalClaims > 0)
    """
    # Load the dataset with mixed type tolerance
    df = pd.read_csv(path, sep="|", low_memory=False)

    # Filter for policies with claims
    claim_data = df[df["TotalClaims"] > 0].copy()

    # Drop irrelevant or high-missing columns
    drop_cols = ["UnderwrittenCoverID", "PolicyID", "TransactionMonth", "Country", "make", "Model"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    claim_data.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Drop completely missing numeric columns
    df = df.dropna(axis=1, how="all")
    claim_data = claim_data.dropna(axis=1, how="all")

    # Impute numeric columns
    num_cols_df = df.select_dtypes(include="number").columns
    num_cols_claim = claim_data.select_dtypes(include="number").columns

    if len(num_cols_df) > 0:
        imputer_df = SimpleImputer(strategy="median")
        df[num_cols_df] = pd.DataFrame(imputer_df.fit_transform(df[num_cols_df]), columns=num_cols_df, index=df.index)

    if len(num_cols_claim) > 0:
        imputer_claim = SimpleImputer(strategy="median")
        claim_data[num_cols_claim] = pd.DataFrame(imputer_claim.fit_transform(claim_data[num_cols_claim]), columns=num_cols_claim, index=claim_data.index)

    # One-hot encode categorical columns
    cat_cols_df = df.select_dtypes(include="object").columns
    cat_cols_claim = claim_data.select_dtypes(include="object").columns

    df = pd.get_dummies(df, columns=cat_cols_df, drop_first=True)
    claim_data = pd.get_dummies(claim_data, columns=cat_cols_claim, drop_first=True)

    return df, claim_data

def prepare_data_for_claim_probability(df):
    """
    Prepares the dataset for predicting the probability of a claim.
    Adds a binary target 'HasClaim' and performs cleaning/encoding similar to the main prep pipeline.

    Args:
        df (pd.DataFrame): Full dataset loaded from load_and_prepare_data()

    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Binary target indicating if a claim was made
    """
    # Create binary target
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    # Drop irrelevant columns
    drop_cols = ["UnderwrittenCoverID", "PolicyID", "TransactionMonth", "Country", "make", "Model"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Drop columns that are entirely missing
    df = df.dropna(axis=1, how="all")

    # Impute numeric columns
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Separate target and features
    X = df.drop(columns=["HasClaim"])
    y = df["HasClaim"]

    return X, y

def prepare_data_for_claim_severity(claim_data):
    """
    Prepares features and target for modeling claim severity.

    Args:
        claim_data (DataFrame): Subset of dataset where TotalClaims > 0.

    Returns:
        X (DataFrame): Features for training severity model.
        y (Series): Target variable (TotalClaims).
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer

    # Separate target
    y = claim_data["TotalClaims"].copy()

    # Drop target and identifiers
    X = claim_data.drop(columns=["TotalClaims"], errors="ignore")

    # Impute numeric features
    num_cols = X.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy="median")
        X[num_cols] = pd.DataFrame(imputer.fit_transform(X[num_cols]), columns=num_cols, index=X.index)

    # One-hot encode categorical features
    cat_cols = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y
