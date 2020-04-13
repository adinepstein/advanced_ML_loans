import pandas as pd
import os

DATA_PATH = r'C:\Users\AdinEpstein\Documents\source\loans_data'
AVG_FED_RATE = 0.9625  # 2007-2014
DROP_COLUMNS = ['emp_title', 'loan_status', 'url', 'loan_status', 'verification_status', 'desc', 'title',
                'total_pymnt_inv', 'out_prncp', 'out_prncp',
                'out_prncp', 'out_prncp_inv', 'total_pymnt', 'verification_status_joint', 'hardship_loan_status',
                'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
                'revol_bal_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
                'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
                'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
                'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog'
                ]
DATE_COLS = ['hardship_start_date', 'hardship_end_date', 'payment_plan_start_date', 'debt_settlement_flag_date',
             'settlement_date']
PARSE_MAP = {'term': {'36 month': 36, '60 month': 60},
             'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': '6'},
             'sub_grade': {'A1': 11, 'A2': 12, 'A3': 13, 'A4': 14, 'A5': 15, 'B1': 21, 'B2': 22, 'B3': 23, 'B4': 24,
                           'B5': 25, 'C1': 31, 'C2': 32, 'C3': 33, 'C4': 34, 'C5': 35, 'D1': 41, 'D2': 42, 'D3': 43,
                           'D4': 44, 'D5': 45, 'E1': 51, 'E2': 52, 'E3': 53, 'E4': 54, 'E5': 55, 'F1': 61, 'F2': 62,
                           'F3': 63, 'F4': 64, 'F5': 65, 'G1': 71, 'G2': 72, 'G3': 73, 'G4': 74, 'G5': 75},
             'pymnt_plan': {'n': 0, 'y': 1},
             'emp_length': {'< 1 year': 0.5, '10+ years': 11, '6 years': 6, '5 years': 5, '8 years': 8, '2 years': 2,
                            '9 years': 9, '1 year': 1, '4 years': 4, 'n/a': 0, '7 years': 7, '3 years': 3},
             'home_ownership': {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'ANY': 3},
             'purpose': {'credit_card': 0, 'debt_consolidation': 1, 'major_purchase': 2, 'small_business': 3,
                         'home_improvement': 4, 'other': 5, 'house': 6, 'medical': 7, 'vacation': 8, 'car': 9,
                         'moving': 10, 'renewable_energy': 11, 'wedding': 12, 'educational': 13},
             'state': {'OH': 0, 'MO': 1, 'SC': 2, 'PA': 3, 'RI': 4, 'NC': 5, 'NY': 6, 'AZ': 7, 'VA': 8, 'KS': 9,
                       'AL': 10, 'NM': 11, 'TX': 12, 'MD': 13, 'WA': 14, 'GA': 15, 'LA': 16, 'IL': 17, 'CO': 18,
                       'FL': 19, 'MI': 20, 'IN': 21, 'TN': 22, 'WI': 23, 'CA': 24, 'VT': 25, 'MA': 26, 'NJ': 27,
                       'OR': 28, 'SD': 29, 'MN': 30, 'DC': 31, 'HI': 32, 'DE': 33, 'NH': 34, 'CT': 35, 'NE': 36,
                       'AR': 37, 'NV': 38, 'MT': 39, 'WV': 40, 'WY': 41, 'OK': 42, 'KY': 43, 'MS': 44, 'UT': 45,
                       'ME': 46, 'ND': 47, 'AK': 48,
                       },
             'initial_list_status': {'w': 0, 'f': 1},
             'application_type': {'Individual': 0, 'Joint App': 1},
             'hardship_type': {'INTEREST ONLY-3 MONTHS DEFERRAL': 1},
             'debt_settlement_flag': {'Y': 1, 'N': 0},
             'settlement_status': {'COMPLETE': 0, 'ACTIVE': 1, 'BROKEN': 2},
             'hardship_status': {'COMPLETE': 0, 'ACTIVE': 1, 'BROKEN': 2},
             'hardship_reason': {'NATURAL_DISASTER': 0, 'DIVORCE': 1, 'EXCESSIVE_OBLIGATIONS': 2, 'DISABILITY': 3,
                                 'UNEMPLOYMENT': 4, 'INCOME_CURTAILMENT': 5, 'REDUCED_HOURS': 6, 'MEDICAL': 7,
                                 'FAMILY_DEATH': 8},

             }


def load_data(folder_path):
    data_dfs = []
    for f in os.listdir(folder_path):
        df = pd.read_csv(os.path.join(folder_path, f))
    all_data = pd.concat(data_dfs)
    return all_data


def parse_data(df):
    pass


def split_data(df):
    pass


def calculate_npv(df):
    pass


def calculate_roi(df):
    pass
