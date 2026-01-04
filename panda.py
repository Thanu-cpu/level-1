import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('customer_data.csv')

# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Fix invalid Name values (names that are just numbers)
# Identify names that are numeric and mark them as 'Unknown'
df.loc[df['Name'].astype(str).str.isnumeric(), 'Name'] = 'Unknown'

# 3. Handle Numerical Data (Including Negative Ages)
# Replace negative ages with NaN so they can be filled with the median
df.loc[df['Age'] < 0, 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

# 4. Correct spelling errors and fill missing categorical values
gender_map = {'Femlae': 'Female', 'mle': 'Male', 'Unknown': 'Other'}
df['Gender'] = df['Gender'].replace(gender_map).fillna('Other')

country_map = {'Indai': 'India', 'Canda': 'Canada'}
df['Country'] = df['Country'].replace(country_map).fillna('Unknown')

device_map = {'dasktop': 'desktop', 'moblie': 'mobile'}
df['PreferredDevice'] = df['PreferredDevice'].replace(device_map).fillna('Unknown')

# 5. Dates and Financials
df['SignupDate'] = pd.to_datetime(df['SignupDate'], errors='coerce')
df['LastLogin'] = pd.to_datetime(df['LastLogin'], errors='coerce')
df['TotalPurchase'] = df['TotalPurchase'].fillna(0)

# 6. Clean Email column
df['Email'] = df['Email'].str.strip().str.lower()
invalid_email = (df['Email'].isna()) | (df['Email'] == '@example.com') | (~df['Email'].str.contains('@', na=False))
df.loc[invalid_email, 'Email'] = 'unknown@example.com'

# 7. Drop rows missing the primary key (CustomerID)
df = df.dropna(subset=['CustomerID'])

# Save the final cleaned file
df.to_csv('cleaned_customer_data_v2.csv', index=False)