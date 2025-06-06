import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Hardcoded email credentials
SENDER_EMAIL = "gokul.220btccse041@sushantuniversity.edu.in"
RECEIVER_EMAIL = "madhavik.agarwal@samarth.community"
EMAIL_PASSWORD = "asxm hriw skph mfpb"  # Hardcoded password

# Decode and write FIREBASE_CREDENTIALS to file
firebase_creds = os.environ["FIREBASE_CREDENTIALS"]
with open("firebase_credentials.json", "w") as f:
    json.dump(json.loads(firebase_creds), f)

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Calculate timestamp for 1 hour ago (UTC)
one_hour_ago = datetime.utcnow() - timedelta(hours=1)

def fetch_accounts():
    """Fetch all account documents with their IDs and names"""
    accounts_ref = db.collection("accounts")
    docs = accounts_ref.stream()
    return [{'id': doc.id, 'name': doc.to_dict().get('name', 'N/A')} for doc in docs]

def fetch_messages(account_id):
    """Fetch recent messages for a specific account"""
    chats_ref = db.collection("chats")
    chat_docs = chats_ref.where('accountId', '==', account_id) \
                         .where('createdAt', '>=', one_hour_ago) \
                         .stream()
    return [{
        'account_id': account_id,
        'message': chat_doc.to_dict().get('message', ''),
        'createdAt': chat_doc.to_dict().get('createdAt').isoformat() if chat_doc.to_dict().get('createdAt') else None
    } for chat_doc in chat_docs]

def fetch_all_messages(account_ids):
    """Parallel message fetching with progress tracking"""
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_messages, acc_id): acc_id for acc_id in account_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching messages"):
            results[futures[future]] = future.result()
    return results

def init_classifier():
    """Initialize the urgency classification model"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def classify_urgency(message, tokenizer, model):
    """Classify message urgency using keywords and sentiment analysis"""
    # Handle empty messages
    if not message or not isinstance(message, str):
        return "NOT URGENT"
    
    message_lower = message.lower()
    urgent_keywords = [
        "urgent", "emergency", "immediately", "asap", "problem", "issue",
        "need help", "need support", "critical", "not working",
        "replacement", "refund", "fail", "crash", "broken",
        "help needed", "important", "priority", "as soon as possible"
    ]
    
    # Keyword-based urgency detection
    if any(kw in message_lower for kw in urgent_keywords):
        return "URGENT"
    
    # Sentiment-based classification
    try:
        inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        # Class 0: Negative, Class 1: Neutral, Class 2: Positive
        return "URGENT" if predicted_class == 0 else "NOT URGENT"
    except Exception as e:
        print(f"Classification error: {e}")
        return "NOT URGENT"

def send_email_with_attachment(csv_path):
    """Send email with CSV attachment using hardcoded credentials"""
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "Urgent Messages Report"
    
    # Email body
    body = f"""
    Automated report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    This report contains messages classified as urgent from the last hour.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach CSV file
    with open(csv_path, "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name=os.path.basename(csv_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_path)}"'
    msg.attach(part)
    
    # Send email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("✅ Report sent successfully via email")
    except Exception as e:
        print(f"❌ Email sending failed: {str(e)}")

def main():
    # Fetch accounts and messages
    accounts = fetch_accounts()
    account_ids = [acc['id'] for acc in accounts]
    all_messages = fetch_all_messages(account_ids)
    
    # Flatten messages and create DataFrame
    messages_list = []
    for account_id, msg_list in all_messages.items():
        messages_list.extend(msg_list)
    
    if not messages_list:
        print("No recent messages found")
        return
    
    df = pd.DataFrame(messages_list)
    
    # Add account names
    account_map = {acc['id']: acc['name'] for acc in accounts}
    df['account_name'] = df['account_id'].map(account_map)
    
    # Initialize classifier
    tokenizer, model = init_classifier()
    
    # Classify urgency
    tqdm.pandas(desc="Classifying urgency")
    df['urgency'] = df['message'].progress_apply(
        lambda msg: classify_urgency(msg, tokenizer, model)
    )
    
    # Save results
    csv_path = 'urgent_messages_report.csv'
    df = df[['account_id', 'account_name', 'message', 'urgency', 'createdAt']]
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Report generated with {len(df)} messages")
    
    # Send email with attachment
    send_email_with_attachment(csv_path)

if __name__ == "__main__":
    main()
