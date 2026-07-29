import pandas as pd
import numpy as np

# =========================================================
# 1. LOAD DATA & APPLY HARD-CODED VARIABLES
# =========================================================
print("Loading cleaned clickstream data...")
# Now grabbing the file that 01.py just cleaned
clickstream = pd.read_csv('cleaned_01_clickstream.csv')
clickstream['eventdate'] = pd.to_datetime(clickstream['eventdate'])

print("Applying Hard Coded URL variables...")
# Media
clickstream['AudioMp3'] = clickstream['path'].str.contains('.mp3|/audio/', case=False, na=False).astype(int)
clickstream['Video'] = clickstream['path'].str.contains('.mp4|/video/', case=False, na=False).astype(int)
clickstream['Download'] = clickstream['path'].str.contains('/download', case=False, na=False).astype(int)

# Document Types
clickstream['LoanEstimateRelated'] = clickstream['path'].str.contains('loanestimate|/LE/', case=False, na=False).astype(int)
clickstream['CDRelated'] = clickstream['path'].str.contains('closingdisclosure|/CD/', case=False, na=False).astype(int)
clickstream['LEDocument'] = clickstream['path'].str.contains('/le_document', case=False, na=False).astype(int)
clickstream['CDDocument'] = clickstream['path'].str.contains('/cd_document', case=False, na=False).astype(int)
clickstream['LoanTermsRelated'] = clickstream['path'].str.contains('/terms', case=False, na=False).astype(int)

# Information Types
clickstream['Personalized'] = clickstream['path'].str.contains('/personalized', case=False, na=False).astype(int)
clickstream['GeneralFinancial'] = clickstream['path'].str.contains('/general', case=False, na=False).astype(int)
clickstream['MortgageRelated'] = clickstream['path'].str.contains('/mortgage', case=False, na=False).astype(int)
clickstream['ProcessRelated'] = clickstream['path'].str.contains('/process', case=False, na=False).astype(int)

# Goals & Roles
clickstream['Goal_to_inform'] = clickstream['path'].str.contains('/inform', case=False, na=False).astype(int)
clickstream['Goal_to_Advise'] = clickstream['path'].str.contains('/advise', case=False, na=False).astype(int)
clickstream['BorrowerMortgageProcessRelated'] = clickstream['path'].str.contains('/borrower_process', case=False, na=False).astype(int)
clickstream['LenderMortgageProcessRelated'] = clickstream['path'].str.contains('/lender_process', case=False, na=False).astype(int)

# =========================================================
# 2. CALCULATED URL VARIABLES
# =========================================================
print("Applying Calculated URL variables...")
clickstream['LEDownload'] = ((clickstream['Download'] == 1) & (clickstream['LoanEstimateRelated'] == 1)).astype(int)
clickstream['CDDownload'] = ((clickstream['Download'] == 1) & (clickstream['CDRelated'] == 1)).astype(int)

# Language Flags (Forward Fill Logic)
clickstream = clickstream.sort_values(by=['user_hash', 'eventdate'])
clickstream['lang_flag'] = np.nan
clickstream.loc[clickstream['path'].str.contains('/translations/en', na=False), 'lang_flag'] = 'English'
clickstream.loc[clickstream['path'].str.contains('/translations/es', na=False), 'lang_flag'] = 'Spanish'

clickstream['lang_flag'] = clickstream.groupby('user_hash')['lang_flag'].ffill()
clickstream['lang_flag'] = clickstream['lang_flag'].fillna('English') # Default

clickstream['English'] = (clickstream['lang_flag'] == 'English').astype(int)
clickstream['Spanish'] = (clickstream['lang_flag'] == 'Spanish').astype(int)
clickstream.drop(columns=['lang_flag'], inplace=True)

# =========================================================
# 3. CALCULATING SESSIONS (15 MINUTE RULE)
# =========================================================
print("Calculating Sessions (Timeout threshold: 15 mins)...")
clickstream['time_spent_on_page_seconds'] = clickstream.groupby('user_hash')['eventdate'].diff().dt.total_seconds().shift(-1)
clickstream['time_spent_on_page_seconds'] = clickstream['time_spent_on_page_seconds'].fillna(60)
clickstream.loc[clickstream['time_spent_on_page_seconds'] > (15 * 60), 'time_spent_on_page_seconds'] = 60

clickstream['time_since_last_click'] = clickstream.groupby('user_hash')['eventdate'].diff().dt.total_seconds()
clickstream['is_new_session'] = (clickstream['time_since_last_click'].isna()) | (clickstream['time_since_last_click'] > (15 * 60))
clickstream['session_id'] = clickstream.groupby('user_hash')['is_new_session'].cumsum()

session_level = clickstream.groupby(['user_hash', 'session_id']).agg(
    session_start_time=('eventdate', 'min'),
    session_end_time=('eventdate', 'max'),
    pages_viewed_in_session=('path', 'count'),
    session_duration_seconds=('time_spent_on_page_seconds', 'sum')
).reset_index()

session_level = session_level.sort_values(by=['user_hash', 'session_start_time'])
session_level['prev_session_end'] = session_level.groupby('user_hash')['session_end_time'].shift(1)
session_level['inter_session_elapsed_time_hrs'] = (session_level['session_start_time'] - session_level['prev_session_end']).dt.total_seconds() / 3600

# =========================================================
# 4. USER LEVEL AGGREGATIONS
# =========================================================
print("Aggregating to User Level Dataset...")

# Extract just the date first to avoid the GroupBy dt error
clickstream['event_date_only'] = clickstream['eventdate'].dt.date

user_summary = clickstream.groupby('user_hash').agg(
    Number_of_webpages_visited=('path', 'count'),
    Number_of_unique_webpages_visited=('path', 'nunique'),
    Number_of_English_webpages_visited=('English', 'sum'),
    Number_of_Spanish_webpages_visited=('Spanish', 'sum'),
    AudioMp3_count=('AudioMp3', 'sum'),
    Video_count=('Video', 'sum'),
    LE_Related_count=('LoanEstimateRelated', 'sum'),
    CD_Related_count=('CDRelated', 'sum'),
    ProcessRelated_count=('ProcessRelated', 'sum'),
    Goal_to_inform_count=('Goal_to_inform', 'sum'),
    Goal_to_Advise_count=('Goal_to_Advise', 'sum'),
    Total_time_on_site_seconds=('time_spent_on_page_seconds', 'sum'),
    First_access_date=('eventdate', 'min'),
    Last_access_date=('eventdate', 'max'),
    Total_days_account_accessed=('event_date_only', 'nunique')
).reset_index()

# Merge session metrics into user summary
session_summaries = session_level.groupby('user_hash').agg(
    Number_of_sessions=('session_id', 'max'),
    Avg_pages_per_session=('pages_viewed_in_session', 'mean'),
    Avg_inter_session_time_hrs=('inter_session_elapsed_time_hrs', 'mean')
).reset_index()

user_summary = pd.merge(user_summary, session_summaries, on='user_hash', how='left')

# =========================================================
# 5. EXPORT
# =========================================================
print("Saving clean data files...")
clickstream.to_csv('processed_02_clickstream_url_level.csv', index=False)
session_level.to_csv('processed_02_clickstream_session_level.csv', index=False)
user_summary.to_csv('processed_02_clickstream_user_level.csv', index=False)
print("Script 02 Complete!")