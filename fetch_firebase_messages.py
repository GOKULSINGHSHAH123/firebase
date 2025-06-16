import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
import openai
import time
import re

# Environment configuration
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "aisamarth2016@gmail.com")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "madhavik.agarwal@samarth.community")
EMAIL_PASSWORD = "xcag hxya fypu nbsq"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
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

def get_ist_time_range():
    """Get the current hour interval in IST (UTC+5:30)"""
    current_utc = datetime.utcnow()
    ist_offset = timedelta(hours=5, minutes=30)
    current_ist = current_utc + ist_offset
    
    # Calculate start and end of current hour in IST
    start_ist = current_ist.replace(minute=0, second=0, microsecond=0)
    end_ist = start_ist + timedelta(hours=1)
    
    # Convert to UTC for Firebase query
    start_utc = start_ist - ist_offset
    end_utc = end_ist - ist_offset
    
    return start_utc, end_utc, start_ist, end_ist

def fetch_accounts():
    """Fetch all account documents with their IDs and names"""
    return [
        {'id': doc.id, 'name': doc.to_dict().get('name', 'N/A')} 
        for doc in db.collection("accounts").stream()
    ]

def fetch_messages(account_id, start_utc, end_utc):
    """Fetch recent messages for a specific account within time range"""
    chats_ref = db.collection("chats")
    query = (chats_ref
             .where('accountId', '==', account_id)
             .where('createdAt', '>=', start_utc)
             .where('createdAt', '<', end_utc))
    
    return [{
        'account_id': account_id,
        'message': doc.to_dict().get('message', ''),
        'createdAt': doc.to_dict().get('createdAt')
    } for doc in query.stream()]

def fetch_all_messages(account_ids, start_utc, end_utc):
    """Parallel message fetching with progress tracking"""
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_messages, acc_id, start_utc, end_utc): acc_id 
                  for acc_id in account_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching messages"):
            results[futures[future]] = future.result()
    return results

def smart_truncate(text, max_length=4000):
    """Ensure text fits within model context limits"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + ' [...]'

def create_batch_prompt(messages):
    num_messages = len(messages)
    prompt = PROMPT_TEMPLATE.format(num_messages=num_messages) + "\n"
    
    for i, msg in enumerate(messages, 1):
        clean_msg = msg.replace('"', "'").strip()
        truncated = smart_truncate(clean_msg)
        prompt += f"\n{i}. {truncated}"
    
    return prompt

def classify_batch(messages, depth=0):
    """Classify message batch using GPT-4o with retry logic"""
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

def send_email_with_attachment(csv_path, time_interval):
    """Send email with CSV attachment using EmailMessage"""
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"Urgent Messages Report - {time_interval} IST"
    
    body = f"""Automated report generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

This report contains messages classified as urgent during:
{time_interval} IST
"""
    msg.set_content(body)
    
    # Add CSV attachment
    with open(csv_path, "rb") as f:
        csv_data = f.read()
    msg.add_attachment(
        csv_data,
        maintype="text",
        subtype="csv",
        filename=os.path.basename(csv_path)
    )
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("aisamarth2016@gmail.com", "xcag hxya fypu nbsq")
            server.send_message(msg)
        print("✅ Report sent successfully via email")
    except Exception as e:
        print(f"❌ Email sending failed: {str(e)}")

def append_to_master_csv(df, filename="all_messages.csv"):
    """Append messages to master CSV file in GitHub Actions"""
    try:
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)
        print(f"✅ Appended {len(df)} messages to master CSV")
    except Exception as e:
        print(f"❌ Error appending to master CSV: {str(e)}")

def main():
    # Get IST time range for current hour
    start_utc, end_utc, start_ist, end_ist = get_ist_time_range()
    time_interval = f"{start_ist.strftime('%H:%M')} - {end_ist.strftime('%H:%M')}"
    print(f"⌛ Processing messages from {time_interval} IST")
    
    # Fetch accounts and messages
    accounts = fetch_accounts()
    account_map = {acc['id']: acc['name'] for acc in accounts}
    all_messages = fetch_all_messages(list(account_map.keys()), start_utc, end_utc)
    
    # Flatten messages
    messages_list = []
    for msg_list in all_messages.values():
        messages_list.extend(msg_list)
    
    if not messages_list:
        print("⏭️ No recent messages found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(messages_list)
    df['account_name'] = df['account_id'].map(account_map)
    
    # Add IST timestamps
    ist_offset = timedelta(hours=5, minutes=30)
    df['createdAt_ist'] = df['createdAt'].apply(
        lambda x: (x + ist_offset).strftime('%Y-%m-%d %H:%M:%S') if x else None
    )
    
    # Classify urgency using GPT-4o
    df['urgency'] = batch_classify_messages(df['message'].tolist())
    
    # Create urgent-only report
    urgent_df = df[df['urgency'] == 'urgent']
    
    # Save and send urgent report
    if not urgent_df.empty:
        csv_filename = f"urgent_messages_{start_ist.strftime('%Y%m%d_%H%M')}.csv"
        urgent_df[['account_id', 'account_name', 'message', 'urgency', 'createdAt_ist']].to_csv(csv_filename, index=False)
        print(f"✅ Generated urgent report with {len(urgent_df)} messages")
        send_email_with_attachment(csv_filename, time_interval)
    else:
        print("⏭️ No urgent messages to report")
    
    # Append all messages to master CSV
    df['createdAt'] = df['createdAt'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else None
    )
    append_to_master_csv(df[['account_id', 'account_name', 'message', 'urgency', 'createdAt', 'createdAt_ist']])

if __name__ == "__main__":
    main()
