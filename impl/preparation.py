"""
Module to prepare data for model training
"""
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from config import *

warnings.filterwarnings("ignore", category=UserWarning)


def convert_sex(val):
    """
    Convert string representation to int. If the value is missing keep it as is.
    :param val: value to be converted
    :return: int or na
    """
    if pd.isna(val):
        return val
    return 1 if val == "M" else 0


def fillna_median(Xtrain, Xval, Xtest):
    """
    Fill missing values in all columns with the median of the column
    :param Xtrain: training data
    :param Xval: evaluation data
    :param Xtest: test data
    :return: modified datasets
    """
    colnames = Xtrain.columns
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    Xtrain = imp_median.fit_transform(Xtrain)
    Xtrain = pd.DataFrame(Xtrain, columns=colnames)
    Xval = imp_median.transform(Xval)
    Xval = pd.DataFrame(Xval, columns=colnames)
    Xtest = imp_median.transform(Xtest)
    Xtest = pd.DataFrame(Xtest, columns=colnames)
    return Xtrain, Xval, Xtest


def load_credit():
    """
    Load the south german credit dataset and return it as a dataframe
    :return: prepared datasets
    """
    columns = ["status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration",
               "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property", "age",
               "other_installment_plans", "housing", "number_credits", "job", "people_liable", "telephone",
               "foreign_worker", "credit_risk"]
    df = pd.read_csv("../data/credit/SouthGermanCredit.asc", delimiter=' ')
    df.columns = columns
    # convert 1 to mean there is risk
    df["credit_risk"] = df["credit_risk"].apply(lambda x: 0 if x == 1 else 1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(columns="credit_risk"), df["credit_risk"], test_size=0.4,
                                                    random_state=RAND_VAL)
    Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.5, random_state=RAND_VAL)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def load_heart():
    """
    Load the heart disease datasets and merge them into one dataframe
    :return: prepared datasets
    """
    columns = ["age", "sex", "chest_pain", "rest_bp", "chol", "fast_bs", "rest_ecg", "max_hr", "ex_angina",
               "st_depression", "st_slope", "flouro_vessels", "thalassemia", "diagnosis"]
    df1 = pd.read_csv("../data/heart/processed.hungarian.data", na_values="?", names=columns)
    df2 = pd.read_csv("../data/heart/processed.cleveland.data", na_values="?", names=columns)
    df3 = pd.read_csv("../data/heart/processed.switzerland.data", na_values="?", names=columns)
    df4 = pd.read_csv("../data/heart/processed.va.data", na_values="?", names=columns)
    df = pd.concat([df1, df2, df3, df4])
    # simplify diagnosis to a binary problem
    df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x >= 1 else 0)
    Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(columns="diagnosis"), df["diagnosis"], test_size=0.4,
                                                    random_state=RAND_VAL)
    Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.5, random_state=RAND_VAL)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def load_heart_nonan():
    """
    Load heart disease dataset where missing data is replaced using median
    :return: prepared datasets
    """
    Xtrain, Xval, Xtest, ytrain, yval, ytest = load_heart()
    Xtrain, Xval, Xtest = fillna_median(Xtrain, Xval, Xtest)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def load_hypothyroid():
    """
    Load the hypothyroid dataset and return it as a dataframe
    :return: prepared datasets
    """
    columns = ["diagnosis", "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
               "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium",
               "goitre", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured", "T4U",
               "FTI_measured", "FTI", "TBG_measured", "TBG"]
    df = pd.read_csv("../data/thyroid/hypothyroid.data", delimiter=',', na_values="?", names=columns)
    # drop columns that only determine if the measurement was taken and keep the actual values if they are available
    df.drop(list(df.filter(regex="measured")), axis=1, inplace=True)
    # convert character into numeric values for simplicity
    df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "hypothyroid" else 0)
    df["sex"] = df["sex"].apply(convert_sex)
    # convert all t f to 1 0
    df = df.replace({"t": 1, "f": 0})
    Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(columns="diagnosis"), df["diagnosis"], test_size=0.4,
                                                    random_state=RAND_VAL, stratify=df[["diagnosis"]])
    Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.5, random_state=RAND_VAL,
                                                stratify=ytest)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def load_hypo_nonan():
    """
    Load hypothyroid dataset where missing numeric data is replaced using median and sex with the most frequent value
    :return: prepared datasets
    """
    Xtrain, Xval, Xtest, ytrain, yval, ytest = load_hypothyroid()
    Xtrain, Xval, Xtest = fillna_median(Xtrain, Xval, Xtest)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def load_euthyroid():
    """
    Load the euthyroid dataset and return it as a dataframe
    :return: prepared datasets
    """
    columns = ["diagnosis", "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
               "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium",
               "goitre", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured", "T4U",
               "FTI_measured", "FTI", "TBG_measured", "TBG"]
    df = pd.read_csv("../data/thyroid/sick-euthyroid.data", delimiter=',', na_values="?", names=columns)
    df.drop(list(df.filter(regex="measured")), axis=1, inplace=True)
    # convert character into numeric values for simplicity
    df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "sick-euthyroid" else 0)
    df["sex"] = df["sex"].apply(convert_sex)
    # convert all t f to 1 0
    df = df.replace({"t": 1, "f": 0})
    Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(columns="diagnosis"), df["diagnosis"], test_size=0.4,
                                                    random_state=RAND_VAL, stratify=df[["diagnosis"]])
    Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.5, random_state=RAND_VAL,
                                                stratify=ytest)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def load_euth_nonan():
    """
    Load euthyroid dataset where missing numeric data is replaced using median and sex with the most frequent value
    :return: prepared datasets
    """
    Xtrain, Xval, Xtest, ytrain, yval, ytest = load_euthyroid()
    Xtrain, Xval, Xtest = fillna_median(Xtrain, Xval, Xtest)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest