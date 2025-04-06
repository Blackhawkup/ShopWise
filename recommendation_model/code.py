import numpy as np
import pandas as pd
import os
from apyori import apriori

print("🔍 Current Working Directory:")
print(os.getcwd())

# Load dataset
file_path = './recommendation_model/Market_Basket_Optimisation.csv'
print(f"📂 Checking file exists at: {file_path} → {os.path.exists(file_path)}")

dataset = pd.read_csv(file_path, header=None)

# Convert to transactions
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(20) if str(dataset.values[i,j]) != 'nan'])

print(f"🛒 Total transactions loaded: {len(transactions)}")

# Run Apriori
rules = list(apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2))

print(f"📊 Total rules generated: {len(rules)}")

# Show sample rule if any
if len(rules) > 0:
    print("✅ First Rule Example:")
    print(rules[0])
else:
    print("⚠️ No rules found. Try lowering support/confidence.")
