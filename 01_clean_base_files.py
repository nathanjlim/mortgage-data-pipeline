import pandas as pd

print("Starting Step 01: Standardizing Identifiers across all 5 source files...")

# =========================================================
# 1. CLEAN CLICKSTREAM (user_hash)
# =========================================================
print("Cleaning clickstream data...")
clickstream = pd.read_csv('talkument_userinteractions(user_usage).csv')
clickstream = clickstream.dropna(subset=['user_hash'])
clickstream['user_hash'] = clickstream['user_hash'].astype(str).str.strip().str.lower()
clickstream.to_csv('cleaned_01_clickstream.csv', index=False)

# =========================================================
# 2. CLEAN USER ACCOUNTS (user_hash)
# =========================================================
print("Cleaning user account data...")
useraccount = pd.read_csv('talkument_useraccount(users).csv')
useraccount = useraccount.dropna(subset=['user_hash'])
useraccount['user_hash'] = useraccount['user_hash'].astype(str).str.strip().str.lower()
useraccount.to_csv('cleaned_01_useraccount.csv', index=False)

# =========================================================
# 3. CLEAN LOAN APPLICANTS CROSSWALK (user_hash & loannumber)
# =========================================================
print("Cleaning loan applicants data...")
loans = pd.read_csv('talkument_loan_applicants(loan_applicants).csv')
loans = loans.dropna(subset=['loannumber'])

# Force ID columns to be identical text to prevent merge crashes
loans['loannumber'] = loans['loannumber'].astype(str).str.strip()
loans['loannumber'] = loans['loannumber'].str.replace(r'\.0$', '', regex=True)

if 'user_hash' in loans.columns:
    loans['user_hash'] = loans['user_hash'].astype(str).str.strip().str.lower()
loans.to_csv('cleaned_01_loans.csv', index=False)

# =========================================================
# 4. CLEAN PILOT BUCKETS (loan_number)
# =========================================================
print("Cleaning pilot buckets data...")
buckets = pd.read_csv('talkument_pilot_buckets(pilot_record).csv')
buckets = buckets.dropna(subset=['loan_number'])
buckets['loan_number'] = buckets['loan_number'].astype(str).str.strip()
buckets['loan_number'] = buckets['loan_number'].str.replace(r'\.0$', '', regex=True)
buckets.to_csv('cleaned_01_buckets.csv', index=False)

# =========================================================
# 5. CLEAN LOAN OUTCOMES (loan_number)
# =========================================================
print("Cleaning loan application outcomes data...")
loan_outcomes = pd.read_csv('loan_application_data_partial_copy.csv')

# Clean column names exactly as you had it
loan_outcomes.columns = loan_outcomes.columns.str.strip().str.lower() 

loan_outcomes = loan_outcomes.dropna(subset=['loan_number'])
loan_outcomes['loan_number'] = loan_outcomes['loan_number'].astype(str).str.strip()
loan_outcomes['loan_number'] = loan_outcomes['loan_number'].str.replace(r'\.0$', '', regex=True)
loan_outcomes.to_csv('cleaned_01_loan_outcomes.csv', index=False)

print("Step 01 Complete! All ID columns are now perfectly matched and saved.")