# Some ideas
import pandas as pd
import numpy as np
import re
EIDL_advance = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL_temp\EIDL_matched_filled.csv")

# removed unwanted columns
colNames = list(EIDL_advance.columns)

# columns we want to keep
colNames.remove('AWARDEEORRECIPIENTLEGALENTITYNAME')
colNames.remove('LEGALENTITYADDRLINE1')
colNames.remove('FACEVALUEOFDIRECTLOANORLOANGUARANTEE')
colNames.remove('firm_id')
colNames.remove('loan_number')

# drop all columns we don't want
EIDL_advance.drop(columns = colNames, inplace = True)

# rename columns
EIDL_advance.rename(columns = {'AWARDEEORRECIPIENTLEGALENTITYNAME': 'EIDL_Name',
                               'LEGALENTITYADDRLINE1': 'EIDL_Address',
                               'FACEVALUEOFDIRECTLOANORLOANGUARANTEE': 'Loan_Amount'}, inplace = True)

# handle abnormalities
EIDL_advance['Loan_Amount'] = abs(EIDL_advance['Loan_Amount'])

# convert to long
EIDL_long = EIDL_advance.pivot(index='firm_id', columns='loan_number', values=['Loan_Amount'])

# change column to normal format
EIDL_long.columns = [f"{Q}_{col}" for col, Q in EIDL_long.columns]

# use regular expression to resort columns
EIDL_long_columns = list(EIDL_long.columns)
sorted_columns = sorted(EIDL_long_columns, key = lambda x: int(re.findall(r'\d+', x)[0]))
EIDL_long = EIDL_long[sorted_columns]

# remove duplicate firm_id to merge back with the long format
EIDL_dedup = EIDL_advance.drop_duplicates(subset = 'firm_id')
EIDL_dedup.drop(columns = ['Loan_Amount', 'loan_number'], inplace = True)

EIDL_merged = EIDL_dedup.merge(EIDL_long, on = ['firm_id'] ,how = 'inner')

for col in list(EIDL_merged.columns):
    try:
        EIDL_merged[col] = EIDL_merged[col].astype("float64")
    except:
        EIDL_merged[col] = EIDL_merged[col].astype(str)

EIDL_merged_copy = EIDL_merged.copy()
# resolve Unicode problem
EIDL_merged_copy['EIDL_Name'] = EIDL_merged_copy['EIDL_Name'].str.replace('[^\x00-\x7F]','')
EIDL_merged_copy['EIDL_Address'] = EIDL_merged_copy['EIDL_Address'].str.replace('[^\x00-\x7F]','')

# EIDL_merged_copy['EIDL_Name'] = EIDL_merged_copy['EIDL_Name'].str.encode('latin-1', 'replace')
# EIDL_merged_copy['EIDL_Address'] = EIDL_merged_copy['EIDL_Address'].str.encode('latin-1', 'replace')

# for col in list(EIDL_merged_copy.columns):
#     try:
#         EIDL_merged_copy[col] = EIDL_merged_copy[col].astype("float64")
#     except:
#         EIDL_merged_copy[col] = EIDL_merged_copy[col].astype(str)

EIDL_merged_copy.to_stata("EIDL_matched_filled.dta", write_index= False, version = 118)