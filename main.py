import pandas as pd

print("Loading data...")
clickstream   = pd.read_csv('talkument_userinteractions(user_usage).csv')
useraccount   = pd.read_csv('talkument_useraccount(users).csv')
loans         = pd.read_csv('talkument_loan_applicants(loan_applicants).csv')
buckets       = pd.read_csv('talkument_pilot_buckets(pilot_record).csv')


print("Processing clickstream behaviors...")

# eventdate recognizes as time
clickstream['eventdate'] = pd.to_datetime(clickstream['eventdate'], format='mixed')

# flag actions (true/false)
clickstream['clicked_audio'] = clickstream['path'].str.contains('/audio/', case=False, na=False)
clickstream['clicked_terms'] = clickstream['path'].str.contains('/AcceptTerms', case=False, na=False)


print("Aggregating data per user...")

user_summary = clickstream.groupby('user_hash').agg(
    total_clicks=('path', 'count'),
    audio_listens=('clicked_audio', 'sum'),
    terms_reached=('clicked_terms', 'sum'),
    first_click=('eventdate', 'min'),
    last_click=('eventdate', 'max')
).reset_index()

# calculate time spent (mins)
user_summary['time_spent_mins'] = (user_summary['last_click'] - user_summary['first_click']).dt.total_seconds() / 60

# drop data for final merge here
user_summary = user_summary.drop(columns=['first_click', 'last_click'])


print("Merging all tables into final dataset...") #MERGE STARTS 

#attach experiment buckets to loans
master_data = pd.merge(loans, buckets, left_on='loannumber', right_on='loan_number', how='left')

#attach user account profiles (language, expertise)
master_data = pd.merge(master_data, useraccount, on='user_hash', how='left')

#attach web behaviors
master_data = pd.merge(master_data, user_summary, on='user_hash', how='left')

#debug test
print("Data pipeline complete! Here are the first 5 rows:")
print(master_data.head())

#print here
master_data.to_csv('master_dataset_results.csv', index=False)
print("Saved final dataset as 'master_dataset_results.csv'")