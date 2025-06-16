import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import openai
import time
import re

# Environment configuration
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "gokulsinghshah041@gmail.com")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "madhavik.agarwal@samarth.community")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "asxm hriw skph mfpb")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Set in environment variables
FIREBASE_CREDENTIALS = os.environ["FIREBASE_CREDENTIALS"]

# Classification constants
BATCH_SIZE = 8
MAX_RETRIES = 3
MAX_DEPTH = 3
MODEL = "gpt-4o"
PROMPT_TEMPLATE = """
CLASSIFY ELDER CARE MESSAGE URGENCY
-----------------------------------
Analyze messages between elderly children and care leaders. Classify each message as:
- 'urgent' if it meets ANY:
  • Requires action/follow-up
  • Shows dissatisfaction/complaints
  • Reports service issues
  • Contains critical health/safety updates
  • Includes customer requests
  • Needs immediate attention

- 'not urgent' for:
  • General updates without concern
  • Positive feedback
  • Routine information

Return EXACTLY {num_messages} classifications as a JSON array: ["urgent", "not urgent", ...]
"""

# Initialize Firebase
with open("firebase_credentials.json", "w") as f:
    json.dump(json.loads(FIREBASE_CREDENTIALS), f)

cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Calculate timestamp for 1 hour ago (UTC)
one_hour_ago = datetime.utcnow() - timedelta(hours=1)

def fetch_accounts():
    """Fetch all account documents with their IDs and names"""
    return [
        {'id': doc.id, 'name': doc.to_dict().get('name', 'N/A')} 
        for doc in db.collection("accounts").stream()
    ]

def fetch_messages(account_id):
    """Fetch recent messages for a specific account"""
    chats_ref = db.collection("chats")
    query = chats_ref.where('accountId', '==', account_id).where('createdAt', '>=', one_hour_ago)
    return [{
        'account_id': account_id,
        'message': doc.to_dict().get('message', ''),
        'createdAt': doc.to_dict().get('createdAt').isoformat() if doc.to_dict().get('createdAt') else None
    } for doc in query.stream()]

def fetch_all_messages(account_ids):
    """Parallel message fetching with progress tracking"""
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_messages, acc_id): acc_id for acc_id in account_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching messages"):
            results[futures[future]] = future.result()
    return results

def smart_truncate(text, max_length=4000):
    """Ensure text fits within model context limits"""
    return text[:max_length] + ' [...]' if len(text) > max_length else text

def classify_batch(messages, depth=0):
    """Classify message batch using GPT-4o with retry logic"""
    if depth > MAX_DEPTH or not messages:
        return ["not urgent"] * len(messages)
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    num_messages = len(messages)
    formatted_messages = [
        f"{i+1}. {smart_truncate(msg.replace('"', "'").strip())}"
        for i, msg in enumerate(messages)
    ]
    
    prompt = PROMPT_TEMPLATE.format(num_messages=num_messages) + "\n" + "\n".join(formatted_messages)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an elder care urgency classification assistant."},
                    {"role": "user", "content": prompt + "\n\nONLY RETURN THE JSON ARRAY. NO OTHER TEXT."}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content)
            if json_match:
                results = json.loads(json_match.group(0))
                if len(results) == num_messages and all(r in ["urgent", "not urgent"] for r in results):
                    return [r.lower() for r in results]
        
        except Exception as e:
            print(f"Classification error (attempt {attempt+1}): {str(e)}")
        
        time.sleep(1.5)
    
    # Recursive batch splitting if retries fail
    if len(messages) > 1:
        mid = len(messages) // 2
        return (classify_batch(messages[:mid], depth + 1) + 
                classify_batch(messages[mid:], depth + 1))
    
    return ["not urgent"]

def batch_classify_messages(messages):
    """Process messages in batches with progress tracking"""
    batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    results = []
    
    for batch in tqdm(batches, desc="Classifying urgency"):
        results.extend(classify_batch(batch))
        time.sleep(1)  # Rate limit protection
    
    return results

def send_email_with_attachment(csv_path):
    """Send email with CSV attachment"""
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "Urgent Messages Report"
    
    body = f"""
    Automated report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    This report contains messages classified as urgent from the last hour.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    with open(csv_path, "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name=os.path.basename(csv_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_path)}"'
    msg.attach(part)
    
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
    account_map = {acc['id']: acc['name'] for acc in accounts}
    all_messages = fetch_all_messages(list(account_map.keys()))
    
    # Flatten messages
    messages_list = []
    for msg_list in all_messages.values():
        messages_list.extend(msg_list)
    
    if not messages_list:
        print("No recent messages found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(messages_list)
    df['account_name'] = df['account_id'].map(account_map)
    
    # Classify urgency using GPT-4o
    df['urgency'] = batch_classify_messages(df['message'].tolist())
    
    # Save and send report
    csv_path = f"urgent_messages_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    df[['account_id', 'account_name', 'message', 'urgency', 'createdAt']].to_csv(csv_path, index=False)
    print(f"✅ Report generated with {len(df)} messages")
    
    # Send email if urgent messages found
    if 'urgent' in df['urgency'].values:
        send_email_with_attachment(csv_path)
    else:
        print("⏭️ No urgent messages to report")

if __name__ == "__main__":
    main()
