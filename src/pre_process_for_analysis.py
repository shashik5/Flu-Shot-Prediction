import csv
import numpy as np


COLOUMN_HEADERS = {"respondent_id": 0, "h1n1_concern": 1, "h1n1_knowledge": 2, "behavioral_antiviral_meds": 3, "behavioral_avoidance": 4, "behavioral_face_mask": 5, "behavioral_wash_hands": 6, "behavioral_large_gatherings": 7, "behavioral_outside_home": 8, "behavioral_touch_face": 9, "doctor_recc_h1n1": 10, "doctor_recc_seasonal": 11, "chronic_med_condition": 12, "child_under_6_months": 13, "health_worker": 14, "health_insurance": 15, "opinion_h1n1_vacc_effective": 16,
                   "opinion_h1n1_risk": 17, "opinion_h1n1_sick_from_vacc": 18, "opinion_seas_vacc_effective": 19, "opinion_seas_risk": 20, "opinion_seas_sick_from_vacc": 21, "age_group": 22, "education": 23, "race": 24, "sex": 25, "income_poverty": 26, "marital_status": 27, "rent_or_own": 28, "employment_status": 29, "hhs_geo_region": 30, "census_msa": 31, "household_adults": 32, "household_children": 33, "employment_industry": 34, "employment_occupation": 35}

BINARY_FIELDS = range(3, 15, 1)


def _parse_binary(code: int):
    return 'Yes' if code == 1 else 'No'


H1N1_CONCERN_CATEGORY = {
    '0': 'Not at all concerned',
    '1': 'Not very concerned',
    '2': 'Somewhat concerned',
    '3': 'Very concerned'
}

H1N1_KNOWLEDGE_CATEGORY = {
    '0': 'No knowledge',
    '1': 'A little knowledge',
    '2': 'A lot of knowledge'
}

opinion_h1n1_vacc_effective = {
    '1': 'Not at all effective',
    '2': 'Not very effective',
    '3': 'Don\'t know',
    '4': 'Somewhat effective',
    '5': 'Very effective'
}

opinion_h1n1_risk = {
    '1': 'Very Low',
    '2': 'Somewhat low',
    '3': 'Don\'t know',
    '4': 'Somewhat high',
    '5': 'Very high'
}

opinion_h1n1_sick_from_vacc = {
    '1': 'Not at all worried',
    '2': 'Not very worried',
    '3': 'Don\'t know',
    '4': 'Somewhat worried',
    '5': 'Very worried'
}

opinion_seas_vacc_effective = {
    '1': 'Not at all effective',
    '2': 'Not very effective',
    '3': 'Don\'t know',
    '4': 'Somewhat effective',
    '5': 'Very effective'
}

opinion_seas_risk = {
    '1': 'Very Low',
    '2': 'Somewhat low',
    '3': 'Don\'t know',
    '4': 'Somewhat high',
    '5': 'Very high'
}

opinion_seas_sick_from_vacc = {
    '1': 'Not at all worried',
    '2': 'Not very worried',
    '3': 'Don\'t know',
    '4': 'Somewhat worried',
    '5': 'Very worried'
}

def _pre_process_row(col_data: str, idx: int):
    if col_data == '':
        return 'Not Answered'

    if idx in BINARY_FIELDS:
        return _parse_binary(int(col_data))

    match idx:
        case 1:
            return H1N1_CONCERN_CATEGORY[col_data]
        case 2:
            return H1N1_KNOWLEDGE_CATEGORY[col_data]
        case 16:
            return opinion_h1n1_vacc_effective[col_data]
        case 17:
            return opinion_h1n1_risk[col_data]
        case 18:
            return opinion_h1n1_sick_from_vacc[col_data]
        case 19:
            return opinion_seas_vacc_effective[col_data]
        case 20:
            return opinion_seas_risk[col_data]
        case 21:
            return opinion_seas_sick_from_vacc[col_data]
        case 32:
            return int(col_data)
        case 33:
            return int(col_data)
        case _:
            return col_data


def pre_process_for_analysis(file_path: str, out_file_path: str):
    info = np.empty((0, 36))
    excluded_respondents = find_no_of_empty_responses_by_respondents(file_path)
    with open(file_path, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            print('Preprocessing record ', reader.line_num)
            if row[0] in excluded_respondents:
                continue
            if (reader.line_num == 1):
                info = np.append(info, np.array(row).reshape((1, len(row))), axis=0)
                continue
            pre_processed_row = list(map(_pre_process_row, row, range(len(row))))
            info = np.append(info, np.array(pre_processed_row).reshape((1, len(row))), axis=0)

    print('New Record Length: ', len(info) - 1)
    with open(out_file_path, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='excel', delimiter=',')
        writer.writerows(info)


def scan_row(row: list[str]):
    no_of_empty = 0
    for c in row:
        if c == '':
            no_of_empty += 1
    return no_of_empty


def find_no_of_empty_responses_by_respondents(file_path: str):
    target_respondent_ids = []
    with open(file_path, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if (reader.line_num == 1):
                continue
            no_of_empty = scan_row(row)
            if(no_of_empty >= 15):
                target_respondent_ids.append(row[0])

    print('Number of respondents with unanswered/empty response greater than or equal to 15 is ', len(target_respondent_ids), ' respondents.')
    return target_respondent_ids