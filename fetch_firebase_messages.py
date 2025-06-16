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
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "aisamarth2016@gmail.com")
RECEIVER_EMAILS = [
    "madhavik.agarwal@samarth.community",
    "asheesh.gupta@samarth.community",
    "arihant.jain@samarth.community"
]
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "oisx vsbh ongm dpke")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FIREBASE_CREDENTIALS = os.environ["FIREBASE_CREDENTIALS"]
MASTER_CSV_PATH = "all_elder_messages.csv"

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

def get_indian_time_range():
    """Get time range for the last full hour in IST"""
    tz_ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(tz_ist)
    
    # Calculate start and end of previous hour
    end_time_ist = now_ist.replace(minute=0, second=0, microsecond=0)
    start_time_ist = end_time_ist - timedelta(hours=1)
    
    # Convert to UTC for Firebase query
    start_time_utc = start_time_ist.astimezone(pytz.utc)
    end_time_utc = end_time_ist.astimezone(pytz.utc)
    
    return start_time_ist, end_time_ist, start_time_utc, end_time_utc

def fetch_accounts():
    """Fetch all account documents with their IDs and names"""
    return [
        {'id': doc.id, 'name': doc.to_dict().get('name', 'N/A')} 
        for doc in db.collection("accounts").stream()
    ]

def fetch_messages(account_id, start_time_utc, end_time_utc):
    """Fetch messages for a specific account within time range"""
    chats_ref = db.collection("chats")
    query = chats_ref.where('accountId', '==', account_id
                          ).where('createdAt', '>=', start_time_utc
                          ).where('createdAt', '<', end_time_utc)
    
    messages = []
    for doc in query.stream():
        data = doc.to_dict()
        messages.append({
            'account_id': account_id,
            'message_id': doc.id,
            'message': data.get('message', ''),
            'createdAt': data.get('createdAt').isoformat() if data.get('createdAt') else None
        })
    return messages

def fetch_all_messages(account_ids, start_time_utc, end_time_utc):
    """Parallel message fetching with progress tracking"""
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for acc_id in account_ids:
            future = executor.submit(
                fetch_messages, 
                acc_id, 
                start_time_utc, 
                end_time_utc
            )
            futures[future] = acc_id
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching messages"):
            acc_id = futures[future]
            results[acc_id] = future.result()
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
                if len(results) == num_messages and all(r.lower() in ["urgent", "not urgent"] for r in results):
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

def update_master_csv(new_df):
    """Update master CSV with all historical data"""
    try:
        if os.path.exists(MASTER_CSV_PATH):
            master_df = pd.read_csv(MASTER_CSV_PATH)
            # Combine and remove duplicates
            combined = pd.concat([master_df, new_df]).drop_duplicates(
                subset=['account_id', 'message_id', 'createdAt']
            )
            combined.to_csv(MASTER_CSV_PATH, index=False)
        else:
            new_df.to_csv(MASTER_CSV_PATH, index=False)
    except Exception as e:
        print(f"Error updating master CSV: {str(e)}")

def send_email_with_attachment(csv_path, time_range):
    """Send email with CSV attachment to multiple recipients"""
    start_str, end_str = time_range
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECEIVER_EMAILS)
    msg['Subject'] = f"Urgent Messages Report ({start_str} to {end_str} IST)"
    
    body = f"""
    Automated report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    This report contains messages classified as urgent from {start_str} to {end_str} IST.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    with open(csv_path, "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name=os.path.basename(csv_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_path)}"'
    msg.attach(part)
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        print("✅ Report sent successfully via email")
    except Exception as e:
        print(f"❌ Email sending failed: {str(e)}")

def main():
    # Get time range for previous hour in IST
    start_ist, end_ist, start_utc, end_utc = get_indian_time_range()
    time_range_str = (
        start_ist.strftime('%H:%M'), 
        end_ist.strftime('%H:%M')
    )
    date_str = start_ist.strftime('%Y%m%d')
    
    print(f"⌚ Processing data for {date_str} {time_range_str[0]}-{time_range_str[1]} IST")
    
    # Fetch accounts and messages
    accounts = fetch_accounts()
    account_map = {acc['id']: acc['name'] for acc in accounts}
    all_messages = fetch_all_messages(
        list(account_map.keys()), 
        start_utc, 
        end_utc
    )
    
    # Flatten messages
    messages_list = []
    for msg_list in all_messages.values():
        messages_list.extend(msg_list)
    
    if not messages_list:
        print("⏭️ No messages found in time range")
        return
    
    # Create DataFrame
    df = pd.DataFrame(messages_list)
    df['account_name'] = df['account_id'].map(account_map)
    
    # Classify urgency using GPT-4o
    df['urgency'] = batch_classify_messages(df['message'].tolist())
    
    # Update master CSV with all messages
    update_master_csv(df[['account_id', 'account_name', 'message_id', 
                         'message', 'urgency', 'createdAt']])
    
    # Filter and save urgent messages
    urgent_df = df[df['urgency'] == 'urgent']
    if not urgent_df.empty:
        # Generate report filename
        report_filename = (
            f"urgent_messages_{date_str}_"
            f"{time_range_str[0].replace(':', '')}-"
            f"{time_range_str[1].replace(':', '')}.csv"
        )
        urgent_df.to_csv(report_filename, index=False)
        print(f"✅ Found {len(urgent_df)} urgent messages")
        
        # Send email with report
        send_email_with_attachment(report_filename, time_range_str)
    else:
        print("⏭️ No urgent messages to report")

if __name__ == "__main__":
    main()
