import pandas as pd
import random
from faker import Faker

fake = Faker()
data = []

buildings = ["Library", "Hostel", "Computer Lab", "Classroom Block", "Admin Office"]

for _ in range(300):
    data.append([
        fake.date_this_year(),
        random.choice(buildings),
        random.randint(50, 500),
        random.choice(["Low", "Medium", "High"])
    ])

df = pd.DataFrame(data, columns=[
    "Date",
    "Building_Name",
    "Energy_Consumption_kWh",
    "Usage_Level"
])

df.to_csv("synthetic_campus_energy_usage.csv", index=False)

print("Dataset created successfully!")
