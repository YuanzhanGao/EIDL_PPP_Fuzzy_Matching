# Task:
# Merge all EIDL dataset through Dec 2020 together and conduct preproessing procedures

import pandas as pd
from rapidfuzz import process, fuzz



EIDL_0401_0609 = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL "
                             r"data\april-2021-delivery-of-eidl-data-through-november-2020"
                             r"\DATAACT_EIDL_LOANS_20200401-20200609.csv", on_bad_lines='skip')

EIDL_0610_0625 = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL "
                             r"data\april-2021-delivery-of-eidl-data-through-november-2020"
                             r"\DATAACT_EIDL_LOANS_20200610-20200625.csv", on_bad_lines='skip')

EIDL_0626_0723 = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL "
                             r"data\april-2021-delivery-of-eidl-data-through-november-2020"
                             r"\DATAACT_EIDL_LOANS_20200626-20200723.csv", on_bad_lines='skip')

EIDL_0724_1115 = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL "
                             r"data\april-2021-delivery-of-eidl-data-through-november-2020"
                             r"\DATAACT_EIDL_LOANS_20200724-20201115.csv", on_bad_lines='skip')

EIDL_rest = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL "
                        r"data\april-2021-delivery-of-eidl-data-through-november-2020\DATAACT_EIDL_LOANS_DMCS2.0.csv"
                        r"", on_bad_lines='skip')

EIDL_2020 = pd.concat([EIDL_0401_0609, EIDL_0610_0625, EIDL_0626_0723, EIDL_0724_1115, EIDL_rest])

# Sort by date
EIDL_2020.sort_values(by=['ACTIONDATE'], ascending=True, inplace=True)

# drop duplicates
EIDL_2020.drop_duplicates(inplace=True)

# remove last row
EIDL_2020.drop([961342], inplace=True)

# reset index
EIDL_2020.reset_index(inplace=True)

# drop the newly added index column
EIDL_2020.drop(columns=['index'], inplace=True)

EIDL_2020['firm_id'] = 0

EIDL_2020.to_csv("EIDL_2020.csv", index=False)
