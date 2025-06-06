import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
import json

# Decode and write FIREBASE_CREDENTIALS to file correctly
firebase_creds = os.environ["FIREBASE_CREDENTIALS"]
with open("firebase_credentials.json", "w") as f:
    json.dump(json.loads(firebase_creds), f)

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Reference to the 'accounts' collection
accounts_ref = db.collection("accounts")

# Stream all documents
docs = accounts_ref.stream()

# Collect document IDs and names
data = []
for doc in docs:
    doc_dict = doc.to_dict()
    name = doc_dict.get('name', 'N/A')  # Default to 'N/A' if name not present
    data.append({'id': doc.id, 'name': name})

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv('messages.csv', index=False)

print("✅ Data fetched and saved to messages.csv")
