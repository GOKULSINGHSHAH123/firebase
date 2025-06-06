import firebase_admin
import os
from firebase_admin import credentials, firestore
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

# Initialize Firebase
with open("firebase_credentials.json", "w") as f:
    f.write(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate("firebase_credentials.json")
# firebase_admin.initialize_app(cred)
db = firestore.client()


one_hour_ago = datetime.utcnow() - timedelta(hours=5)

# Fetch account data
def fetch_accounts():
    accounts_ref = db.collection("accounts")
    docs = accounts_ref.stream()
    return [{'id': doc.id, 'name': doc.to_dict().get('name', 'N/A')} for doc in docs]

# Fetch recent messages for an account
def fetch_messages(account_id):
    chats_ref = db.collection("chats")
    chat_docs = chats_ref.where('accountId', '==', account_id) \
                         .where('createdAt', '>=', one_hour_ago) \
                         .stream()
    return [{
        'message': chat_doc.to_dict().get('message', ''),
        'createdAt': chat_doc.to_dict().get('createdAt').isoformat() if chat_doc.to_dict().get('createdAt') else None
    } for chat_doc in chat_docs]

# Parallel message fetching
def fetch_all_messages(account_ids):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_messages, acc_id): acc_id for acc_id in account_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching messages"):
            results[futures[future]] = future.result()
    return results



# Classify message urgency


accounts = fetch_accounts()
account_ids = [acc['id'] for acc in accounts]
messages = fetch_all_messages(account_ids)

    # Create DataFrame
df = pd.DataFrame(accounts)
df['messages'] = df['id'].map(messages)

# Filter and explode messages
df = df[df['messages'].apply(len) > 0]
df_exploded = df.explode('messages', ignore_index=True)
df_exploded = pd.concat([
        df_exploded.drop(columns=['messages']),
        pd.json_normalize(df_exploded['messages'])
], axis=1)
df_exploded.to_csv("firebase_messages_output.csv", index=False)
