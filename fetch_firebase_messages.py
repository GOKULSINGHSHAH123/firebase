import firebase_admin
import os
from firebase_admin import credentials, firestore
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

# Save FIREBASE_CREDENTIALS from env to file
with open("firebase_credentials.json", "w") as f:
    f.write(os.environ["FIREBASE_CREDENTIALS"])

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")

# Initialize the app
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Just confirming it worked
print("Firebase credentials dumped and Firestore client initialized.")
