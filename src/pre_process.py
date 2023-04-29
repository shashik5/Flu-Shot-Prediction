import csv
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


COLOUMN_HEADERS = {"respondent_id": 0, "h1n1_concern": 1, "h1n1_knowledge": 2, "behavioral_antiviral_meds": 3, "behavioral_avoidance": 4, "behavioral_face_mask": 5, "behavioral_wash_hands": 6, "behavioral_large_gatherings": 7, "behavioral_outside_home": 8, "behavioral_touch_face": 9, "doctor_recc_h1n1": 10, "doctor_recc_seasonal": 11, "chronic_med_condition": 12, "child_under_6_months": 13, "health_worker": 14, "health_insurance": 15, "opinion_h1n1_vacc_effective": 16,
                   "opinion_h1n1_risk": 17, "opinion_h1n1_sick_from_vacc": 18, "opinion_seas_vacc_effective": 19, "opinion_seas_risk": 20, "opinion_seas_sick_from_vacc": 21, "age_group": 22, "education": 23, "race": 24, "sex": 25, "income_poverty": 26, "marital_status": 27, "rent_or_own": 28, "employment_status": 29, "hhs_geo_region": 30, "census_msa": 31, "household_adults": 32, "household_children": 33, "employment_industry": 34, "employment_occupation": 35}

AGE_GROUP_CATEGORY = {
    "18 - 34 Years": 0,
    "35 - 44 Years": 1,
    "45 - 54 Years": 2,
    "55 - 64 Years": 3,
    "65+ Years": 4
}

EDUCATION_CATEGORY = {
    "< 12 Years": 0,
    "12 Years": 1,
    "Some College": 2,
    "College Graduate": 3,
    "": -1
}

RACE_CATEGORY = {
    "White": 0,
    "Black": 1,
    "Other or Multiple": 2,
    "Hispanic": 3
}

SEX_CATEGORY = {
    "Male": 0,
    "Female": 1
}

INCOME_POVERTY_CATEGORY = {
    "Below Poverty": 0,
    "<= $75,000, Above Poverty": 1,
    "> $75,000": 2,
    "": -1
}

MARITIAL_STATUS_CATEGORY = {
    "Not Married": 0,
    "Married": 1,
    "": -1
}

HOUSING_CATEGORY = {
    "Rent": 0,
    "Own": 1,
    "": -1
}

EMPLOYMENT_STATUS_CATEGORY = {
    "Not in Labor Force": 0,
    "Employed": 1,
    "Unemployed": 2,
    "": -1
}

REGION_CATEGORY = {
    "oxchjgsf": 0,
    "bhuqouqj": 1,
    "qufhixun": 2,
    "lrircsnp": 3,
    "atmpeygn": 4,
    "lzgpxyit": 5,
    "fpwskwrf": 6,
    "mlyzmhmf": 7,
    "dqpwygqj": 8,
    "kbazzjca": 9
}

MSA_CATEGORY = {
    "Non-MSA": 0,
    "MSA, Not Principle  City": 1,
    "MSA, Principle City": 2
}

EMPLOYMENT_INDUSTRY_CATEGORY = {
    "pxcmvdjn": 0,
    "rucpziij": 1,
    "wxleyezf": 2,
    "saaquncn": 3,
    "xicduogh": 4,
    "ldnlellj": 5,
    "wlfvacwt": 6,
    "nduyfdeo": 7,
    "fcxhlnwr": 8,
    "vjjrobsf": 9,
    "arjwrbjb": 10,
    "atmlpfrs": 11,
    "msuufmds": 12,
    "xqicxuve": 13,
    "phxvnwax": 14,
    "dotnnunm": 15,
    "mfikgejo": 16,
    "cfqqtusy": 17,
    "mcubkhph": 18,
    "haxffmxo": 19,
    "qnlwzans": 20
}

EMPLOYMENT_OCCUPATION_CATEGORY = {
    "xgwztkwe": 0,
    "xtkaffoo": 1,
    "emcorrxb": 2,
    "vlluhbov": 3,
    "xqwwgdyp": 4,
    "ccgxvspp": 5,
    "qxajmpny": 6,
    "kldqjyjy": 7,
    "mxkfnird": 8,
    "hfxkjkmi": 9,
    "bxpfxfdn": 10,
    "ukymxvdu": 11,
    "cmhcxjea": 12,
    "haliazsg": 13,
    "dlvbwzss": 14,
    "xzmlyyjv": 15,
    "oijqvulv": 16,
    "rcertsgn": 17,
    "tfqavkke": 18,
    "hodpvpew": 19,
    "uqqtjvyb": 20,
    "pvmttkik": 21,
    "dcjcmpih": 22
}


def _pre_process_row(colData: str, idx: int):
    match idx:
        case 22:
            return AGE_GROUP_CATEGORY[colData]
        case 23:
            return EDUCATION_CATEGORY[colData]
        case 24:
            return RACE_CATEGORY[colData]
        case 25:
            return SEX_CATEGORY[colData]
        case 26:
            return INCOME_POVERTY_CATEGORY[colData]
        case 27:
            return MARITIAL_STATUS_CATEGORY[colData]
        case 28:
            return HOUSING_CATEGORY[colData]
        case 29:
            return EMPLOYMENT_STATUS_CATEGORY[colData]
        case 30:
            return REGION_CATEGORY[colData]
        case 31:
            return MSA_CATEGORY[colData]
        case 34:
            return EMPLOYMENT_INDUSTRY_CATEGORY[colData] if (colData in EMPLOYMENT_INDUSTRY_CATEGORY.keys()) else 21
        case 35:
            return EMPLOYMENT_OCCUPATION_CATEGORY[colData] if (colData in EMPLOYMENT_OCCUPATION_CATEGORY.keys()) else 23
        case _:
            return int(colData) if colData else -1


def pre_process(trainingDataFilePath: str, trainingLabelFilePath: str, testingDataFilePath: str):
    # excluded_respondents = _find_no_of_empty_responses_by_respondents(trainingDataFilePath)
    excluded_respondents = []
    return _pre_process_data_file(trainingDataFilePath, excluded_respondents)[:, 1:], _pre_process_labels(trainingLabelFilePath, excluded_respondents)[:, 1:3], _pre_process_data_file(testingDataFilePath, [])[:, 1:]


def _pre_process_data_file(filePath: str, excludedRespondents: list = []):
    info = np.empty((0, 36))
    with open(filePath, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if (reader.line_num == 1 or row[0] in excludedRespondents):
                continue
            pre_processed_row = list(map(_pre_process_row, row, range(len(row))))
            info = np.append(info, np.array(pre_processed_row).reshape((1, len(row))), axis=0)
    return info


def _pre_process_label_row(colData: str, idx: int):
    return int(colData)


def _pre_process_labels(filePath: str, excludedRespondents: list = []):
    labels = np.empty((0, 3))
    with open(filePath, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if (reader.line_num == 1 or row[0] in excludedRespondents):
                continue
            pre_processed_row = list(map(_pre_process_label_row, row, range(len(row))))
            labels = np.append(labels, np.array(pre_processed_row).reshape((1, len(row))), axis=0)
    return labels


def _scan_row(row: list[str]):
    no_of_empty = 0
    for c in row:
        if c == '':
            no_of_empty += 1
    return no_of_empty


def _find_no_of_empty_responses_by_respondents(filePath: str):
    target_respondent_ids = []
    with open(filePath, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if (reader.line_num == 1):
                continue
            no_of_empty = _scan_row(row)
            if (no_of_empty >= 15):
                target_respondent_ids.append(row[0])

    print('Number of respondents with unanswered/empty response greater than or equal to 15 is ', len(target_respondent_ids), 'respondents.')
    return target_respondent_ids


def pre_process_using_pandas(trainingDataFilePath: str, trainingLabelFilePath: str, testingDataFilePath: str):
    features_dataset = pd.read_csv(trainingDataFilePath, index_col="respondent_id")
    labels_dataset = pd.read_csv(trainingLabelFilePath, index_col="respondent_id")
    test_dataset = pd.read_csv(testingDataFilePath, index_col="respondent_id")

    x_train = features_dataset.iloc[:, :].values
    x_test = test_dataset.iloc[:, :].values

    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(x_train[:, :])
    x_train[:, :] = imputer.transform(x_train[:, :])

    imputer.fit(x_test[:, :])
    x_test[:, :] = imputer.transform(x_test[:, :])

    ct = ColumnTransformer(transformers=[('encoder', OrdinalEncoder(), [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34])], remainder='passthrough')
    x_train = np.array(ct.fit_transform(x_train))
    x_test = np.array(ct.fit_transform(x_test))

    return x_train, labels_dataset.iloc[:, :].values, x_test
