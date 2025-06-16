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
import pytz

# Environment configuration
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "gokulsinghshah041@gmail.com")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "madhavik.agarwal@samarth.community")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "asxm hriw skph mfpb")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FIREBASE_CREDENTIALS = os.environ["FIREBASE_CREDENTIALS"]

# Constants
BATCH_SIZE = 8
MAX_RETRIES = 3
MAX_DEPTH = 3
MODEL = "gpt-4o"
ALL_CSV_PATH = "all_classified_messages.csv"

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

# Get last full hour interval in UTC from IST
def get_utc_interval_last_hour():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    last_hour = now_ist.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start_utc = last_hour.astimezone(pytz.utc)
    end_utc = (last_hour + timedelta(hours=1)).astimezone(pytz.utc)
    return start_utc, end_utc, last_hour.strftime('%I %p') + " to " + (last_hour + timedelta(hours=1)).strftime('%I %p')

# Fetch accounts
def fetch_accounts():
    return [{'id': doc.id, 'name': doc.to_dict().get('name', 'N/A')} for doc in db.collection("accounts").stream()]

# Fetch messages in 1-hour interval
def fetch_messages(account_id, start_utc, end_utc):
    query = db.collection("chats").where('accountId', '==', account_id).where('createdAt', '>=', start_utc).where('createdAt', '<', end_utc)
    return [{
        'account_id': account_id,
        'message': doc.to_dict().get('message', ''),
        'createdAt': doc.to_dict().get('createdAt').isoformat() if doc.to_dict().get('createdAt') else None
    } for doc in query.stream()]

# Fetch all in parallel
def fetch_all_messages(account_ids, start_utc, end_utc):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_messages, acc_id, start_utc, end_utc): acc_id for acc_id in account_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching messages"):
            results[futures[future]] = future.result()
    return results

# Prompt creation
def smart_truncate(text, max_length=4000):
    return text if len(text) <= max_length else text[:max_length] + ' [...]'

def create_batch_prompt(messages):
    num_messages = len(messages)
    prompt = PROMPT_TEMPLATE.format(num_messages=num_messages) + "\n"
    for i, msg in enumerate(messages, 1):
        clean_msg = msg.replace('"', "'").strip()
        prompt += f"\n{i}. {smart_truncate(clean_msg)}"
    return prompt

# Classification
def classify_batch(messages, depth=0):
    if depth > MAX_DEPTH or not messages:
        return ["not urgent"] * len(messages)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    num_messages = len(messages)
    prompt = create_batch_prompt(messages)

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

    if len(messages) > 1:
        mid = len(messages) // 2
        return classify_batch(messages[:mid], depth + 1) + classify_batch(messages[mid:], depth + 1)

    return ["not urgent"]

def batch_classify_messages(messages):
    batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    results = []
    for batch in tqdm(batches, desc="Classifying urgency"):
        results.extend(classify_batch(batch))
        time.sleep(1)
    return results

# Email
def send_email_with_attachment(csv_path, interval_text):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"Urgent Messages Report ({interval_text} IST)"

    body = f"Automated urgent message report for the interval {interval_text} IST."
    msg.attach(MIMEText(body, 'plain'))

    with open(csv_path, "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name=os.path.basename(csv_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_path)}"'
    msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("✅ Email with urgent report sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Main
def main():
    # Calculate previous full hour interval in UTC and IST label
    now = datetime.utcnow()
    end_utc = now.replace(minute=0, second=0, microsecond=0)
    start_utc = end_utc - timedelta(hours=1)
    ist_start = (start_utc + timedelta(hours=5, minutes=30)).strftime('%I:%M %p')
    ist_end = (end_utc + timedelta(hours=5, minutes=30)).strftime('%I:%M %p')
    interval_text = f"{ist_start} to {ist_end} IST"

    # Fetch and classify
    accounts = fetch_accounts()
    account_map = {acc['id']: acc['name'] for acc in accounts}
    all_messages = fetch_all_messages(list(account_map.keys()), start_utc, end_utc)
    
    # Flatten
    messages_list = [msg for sublist in all_messages.values() for msg in sublist]
    if not messages_list:
        print("No messages found in the interval.")
        return
    
    df = pd.DataFrame(messages_list)
    df['account_name'] = df['account_id'].map(account_map)
    df['urgency'] = batch_classify_messages(df['message'].tolist())

    # ✅ Append to full history CSV (tracked in GitHub)
    all_csv_path = "all_classified_messages.csv"
    if os.path.exists(all_csv_path):
        df.to_csv(all_csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(all_csv_path, index=False)

    # 📧 Email only urgent messages
    urgent_df = df[df['urgency'] == 'urgent']
    if not urgent_df.empty:
        urgent_csv = f"urgent_messages_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
        urgent_df[['account_id', 'account_name', 'message', 'urgency', 'createdAt']].to_csv(urgent_csv, index=False)
        send_email_with_attachment(urgent_csv, interval_text)
    else:
        print("No urgent messages to report.")


if __name__ == "__main__":
    main()
