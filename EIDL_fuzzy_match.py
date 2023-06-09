# Task:
# Conduct fuzzy matching
import re

import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from ftfy import fix_text
import re
import numpy as np
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process, fuzz


def ngrams(string, n=3):
    string = fix_text(string)
    string = string.encode("ascii", errors='ignore').decode()
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()
    string = re.sub(' +', ' ', string).strip()
    string = ' ' + string + ' '
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def get_matches_df(sparse_matrix, messy_name_vector, clean_name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()

    # sparserows == sparsecols
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    messy_side = np.empty([nr_matches], dtype=object)
    clean_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        messy_side[index] = messy_name_vector[sparserows[index]]
        clean_side[index] = clean_name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'EIDL_side': messy_side,
                         'PPP_side': clean_side,
                         'similairity': similairity})


def address_matching(sparse_matrix, messy_address_vector, clean_address_vector, EIDL_subset, PPP_subset):
    non_zeros = sparse_matrix.nonzero()

    # sparserows == sparsecols
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    for i in range(len(sparserows)):
        messy_address_name = messy_address_vector[
            sparserows[i]]  # get the address of the row that has similar names with PPP's firm names
        clean_address_name = clean_address_vector[sparsecols[i]]

        # print("Hello World")

        if fuzz.ratio(messy_address_name, clean_address_name) > 70:  # if name matches, we assign firm_id
            EIDL_subset['firm_id'][sparserows[i]] = PPP_subset['firm_id'][sparsecols[i]]


# get the selected columns of PPP (borrowername, borrower address, state, firm_id)
PPP_selected_columns = pd.read_stata(r"C:\Users\james\Desktop\Econ_Research\EIDL data\PPP_selected_columns.dta")
PPP_selected_no_dup = PPP_selected_columns.drop_duplicates(subset='firm_id')
# drop nan
PPP_selected_no_dup.drop([4174169], inplace=True)

PPP_state_list = list(PPP_selected_no_dup['borrowerstate'].unique())

# load EIDL dataset
EIDL_2020 = pd.read_csv(r"C:\Users\james\Desktop\Econ_Research\EIDL_2020.csv")

# EIDL_2020_duprows = EIDL_2020[EIDL_2020[['AWARDEEORRECIPIENTLEGALENTITYNAME', 'LEGALENTITYADDRLINE1']].duplicated(keep = False) == True]

# fuzzy matching by state
state_list = list(EIDL_2020['LEGALENTITYSTATECD'].unique())

EIDL_matched = pd.DataFrame()
# testing different fuzzy matching techniques
for state in state_list[1:]:
    print(f"Processing state {state} ...")
    # get subbset
    cali_subset_PPP = PPP_selected_no_dup.loc[PPP_selected_no_dup['borrowerstate'] == state]
    cali_subset_PPP_indexed = cali_subset_PPP.reset_index()
    cali_subset_EIDL = EIDL_2020.loc[EIDL_2020['LEGALENTITYSTATECD'] == state]
    cali_subset_EIDL_indexed = cali_subset_EIDL.reset_index()

    # Get the company name column
    cali_name_col_EIDL = cali_subset_EIDL['AWARDEEORRECIPIENTLEGALENTITYNAME'].values.astype('U')
    cali_name_col_PPP = cali_subset_PPP['borrowername'].values.astype('U')

    cali_address_col_EIDL = cali_subset_EIDL['LEGALENTITYADDRLINE1'].values.astype('U')
    cali_address_col_PPP = cali_subset_PPP['borroweraddress'].values.astype('U')

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix_EIDL = vectorizer.fit_transform(cali_name_col_EIDL)
    tf_idf_matrix_PPP = vectorizer.transform(cali_name_col_PPP)

    matches = awesome_cossim_topn(tf_idf_matrix_EIDL, tf_idf_matrix_PPP.transpose(), 5, 0.9, use_threads=True, n_jobs=4)

    print(f"Matching for state {state} is completed!")

    # save the matches sparse matrix to a temp folder
    scipy.sparse.save_npz(f"EIDL_temp/{state}_name_fuzzyMatching.npz", matches)

    # matches_df = get_matches_df(matches, cali_name_col_EIDL, cali_name_col_PPP, top=None)
    # matches_df = matches_df[matches_df['similairity'] < 0.99999]  # Remove all exact matches

    address_matching(matches, cali_address_col_EIDL, cali_address_col_PPP, cali_subset_EIDL_indexed,
                     cali_subset_PPP_indexed)

    cali_subset_EIDL_indexRemoved = cali_subset_EIDL_indexed.drop(columns = ['index'])

    EIDL_matched = pd.concat([EIDL_matched, cali_subset_EIDL_indexRemoved])

    print(f"Writing data for state {state} ...")
    EIDL_matched.to_csv("EIDL_temp/EIDL_matched.csv", index = False)



# store the store matrix for future use
# scipy.sparse.save_npz('California_name_fuzzyMatching.npz', matches)
#
# matches_loaded = scipy.sparse.load_npz(
#     r"C:\Users\james\Desktop\Econ_Research\EIDL data\temp\California_name_fuzzyMatching.npz")

# name_choice = list(state_subset['AWARDEEORRECIPIENTLEGALENTITYNAME'])
#
# firm_id = 1
# # name fuzzy matching
# for i in range(len(name_choice)):
#     print(i)
#     # company of interest
#
#     company_OI = state_subset.loc[state_subset['AWARDEEORRECIPIENTLEGALENTITYNAME'] == name_choice[i]].iloc[0]
#
#     # only do this if the 'firm_id' column value is 0 (i.e. has not been assigned a firm id yet)
#     if (company_OI['firm_id'] == 0).any():
#         # assign with the current firm id
#
#         company_OI['firm_id'] = firm_id
#         # name fuzzing match
#         extraction = process.extract(name_choice[i], name_choice, scorer=fuzz.WRatio)
#         for name_fuzz_match in extraction:
#             name_matching_comps = state_subset.loc[
#                 state_subset['AWARDEEORRECIPIENTLEGALENTITYNAME'] == name_fuzz_match[0]]
#             # print(len(name_matching_comps))
#             for j in range(len(name_matching_comps)):
#                 if fuzz.ratio(company_OI['LEGALENTITYADDRLINE1'],
#                               name_matching_comps.iloc[j]['LEGALENTITYADDRLINE1']) >= 92:
#                     name_matching_comps.iloc[j]['firm_id'] = firm_id
#                     # print("Completed!")
#
#         firm_id += 1


# nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tf_idf_matrix_PPP)
#
# unique_EIDL_company = set(cali_name_col_EIDL)
#
#
# def getNearestN(query):
#     queryTFIDF_ = vectorizer.transform(query)
#     distances, indices = nbrs.kneighbors(queryTFIDF_)
#     return distances, indices
#
#
# distances, indices = getNearestN(unique_EIDL_company)

