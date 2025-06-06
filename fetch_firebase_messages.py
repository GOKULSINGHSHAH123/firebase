import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta
import os

# Save credentials from environment to a file
with open("firebase_credentials.json", "w") as f:
    f.write(os.environ["FIREBASE_CREDENTIALS"])

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Reference to the 'accounts' collection
accounts_ref = db.collection("accounts")

# Stream all documents
docs = accounts_ref.stream()

# Extract id and name from each document
data = []
for doc in docs:
    doc_dict = doc.to_dict()
    name = doc_dict.get('name', 'N/A')
    data.append({'id': doc.id, 'name': name})

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('messages.csv', index=False)
print("messages.csv saved.")
