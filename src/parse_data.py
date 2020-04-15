import math

import pandas as pd
import os

DATA_PATH = r'C:\Users\AdinEpstein\Documents\source\loans_data'
AVG_FED_RATE = 0.009625  # 2007-2014
DROP_COLUMNS = ['emp_title', 'loan_status', 'url', 'verification_status', 'desc', 'title', 'pymnt_plan',
                'total_pymnt_inv', 'out_prncp', 'out_prncp', 'earliest_cr_line', 'member_id', 'issue_d',
                'out_prncp', 'out_prncp_inv', 'total_pymnt', 'verification_status_joint', 'hardship_loan_status',
                'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                'collection_recovery_fee', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
                'revol_bal_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
                'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
                'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
                'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog', 'hardship_start_date',
                'hardship_end_date', 'payment_plan_start_date', 'debt_settlement_flag_date',
                'settlement_date', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'debt_settlement_flag',
                'open_acc_6m', 'open_act_il', 'open_il_12m',
                'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m',
                'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'policy_code', 'annual_inc_joint',
                'dti_joint', 'verification_status_joint', 'hardship_flag', 'application_type']

PARSE_MAP = {'term': {' 36 months': 36, ' 60 months': 60},
             'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7},
             'sub_grade': {'A1': 11, 'A2': 12, 'A3': 13, 'A4': 14, 'A5': 15, 'B1': 21, 'B2': 22, 'B3': 23, 'B4': 24,
                           'B5': 25, 'C1': 31, 'C2': 32, 'C3': 33, 'C4': 34, 'C5': 35, 'D1': 41, 'D2': 42, 'D3': 43,
                           'D4': 44, 'D5': 45, 'E1': 51, 'E2': 52, 'E3': 53, 'E4': 54, 'E5': 55, 'F1': 61, 'F2': 62,
                           'F3': 63, 'F4': 64, 'F5': 65, 'G1': 71, 'G2': 72, 'G3': 73, 'G4': 74, 'G5': 75},
             'emp_length': {'< 1 year': 0.5, '10+ years': 11, '6 years': 6, '5 years': 5, '8 years': 8, '2 years': 2,
                            '9 years': 9, '1 year': 1, '4 years': 4, 'n/a': 0, '7 years': 7, '3 years': 3},
             'home_ownership': {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'ANY': 3, 'OTHER': 4, 'NONE': 5},
             'purpose': {'credit_card': 0, 'debt_consolidation': 1, 'major_purchase': 2, 'small_business': 3,
                         'home_improvement': 4, 'other': 5, 'house': 6, 'medical': 7, 'vacation': 8, 'car': 9,
                         'moving': 10, 'renewable_energy': 11, 'wedding': 12, 'educational': 13},
             'addr_state': {'OH': 0, 'MO': 1, 'SC': 2, 'PA': 3, 'RI': 4, 'NC': 5, 'NY': 6, 'AZ': 7, 'VA': 8, 'KS': 9,
                            'AL': 10, 'NM': 11, 'TX': 12, 'MD': 13, 'WA': 14, 'GA': 15, 'LA': 16, 'IL': 17, 'CO': 18,
                            'FL': 19, 'MI': 20, 'IN': 21, 'TN': 22, 'WI': 23, 'CA': 24, 'VT': 25, 'MA': 26, 'NJ': 27,
                            'OR': 28, 'SD': 29, 'MN': 30, 'DC': 31, 'HI': 32, 'DE': 33, 'NH': 34, 'CT': 35, 'NE': 36,
                            'AR': 37, 'NV': 38, 'MT': 39, 'WV': 40, 'WY': 41, 'OK': 42, 'KY': 43, 'MS': 44, 'UT': 45,
                            'ME': 46, 'ND': 47, 'AK': 48, 'IA': 49, 'ID': 50
                            },
             'initial_list_status': {'w': 0, 'f': 1},
             'hardship_type': {'INTEREST ONLY-3 MONTHS DEFERRAL': 1},
             'settlement_status': {'COMPLETE': 0, 'ACTIVE': 1, 'BROKEN': 2},
             'hardship_status': {'COMPLETE': 0, 'ACTIVE': 1, 'BROKEN': 2, 'COMPLETED': 0},
             'hardship_reason': {'NATURAL_DISASTER': 0, 'DIVORCE': 1, 'EXCESSIVE_OBLIGATIONS': 2, 'DISABILITY': 3,
                                 'UNEMPLOYMENT': 4, 'INCOME_CURTAILMENT': 5, 'REDUCED_HOURS': 6, 'MEDICAL': 7,
                                 'FAMILY_DEATH': 8},

             }


def load_data(folder_path):
    data_dfs = []
    for f in os.listdir(folder_path):
        df = pd.read_csv(os.path.join(folder_path, f))
        data_dfs.append(df)
    all_data = pd.concat(data_dfs)
    return all_data


def replace_special_values(df):
    df['zip_code'] = df['zip_code'].astype(str).apply(lambda x: float(x.replace('xx', '')) if type(x) == str else -1)
    df['revol_util'] = df['revol_util'].astype(str).apply(lambda x: float(x.replace('%', '')))
    df['int_rate'] = df['int_rate'].apply(lambda x: float(x.replace('%', '')) / 100 if type(x) == str else x / 100)
    df = df.loc[~df.grade.isnull()]
    return df


def parse_data(df):
    df.replace(PARSE_MAP, inplace=True)
    df = replace_special_values(df)
    return df


def split_data(df):
    pass


def calculate_npv(df):
    pass


def calculate_roi(df):
    df['interest_returned'] = df['total_pymnt'] / df['loan_amnt']
    df['loan_years'] = df['term'] / 12
    df['denominator'] = df['loan_years'].apply(lambda x: math.pow(1 + AVG_FED_RATE, x))
    df['roi'] = df['interest_returned'] / df['denominator']
    return df


def get_parsed_data():
    df = load_data(DATA_PATH)
    df = parse_data(df)
    df = calculate_roi(df)
    df.drop(DROP_COLUMNS, axis=1, inplace=True)
    df=df.fillna(-1)
    return df


if __name__ == '__main__':
    path = r'C:\Users\AdinEpstein\Documents\source\loans_data'
    df = load_data(path)
    df = parse_data(df)
    df = calculate_roi(df)
    df[['id', 'total_pymnt', 'loan_amnt', 'interest_returned', 'denominator', 'roi']].to_csv(
        r'C:\Users\AdinEpstein\Documents\data\results\vvv1.csv')
    df.drop(DROP_COLUMNS, axis=1, inplace=True)
    df = df.fillna(-1)
