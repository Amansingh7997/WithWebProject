import praw
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

import praw

reddit = praw.Reddit(
    client_id='okx4v6PVXz86ilMbsJpC8Q',
    client_secret='bxSoHcTS4ci9XY58A9iLtGOrUDYy0g',
    user_agent='BurnoutDetectionApp by /u/Strange-Building-953'
)
# Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Use specific origins in production
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
# Quick test
subreddit = reddit.subreddit("all")
for submission in subreddit.hot(limit=5):
    print(submission.title)



# Define the subreddit and search query
subreddit = reddit.subreddit("all")  # or specify a particular subreddit like 'careerguidance'
query = "burnout work OR workplace"

# Collect posts
posts = []
for submission in subreddit.search(query, limit=100):  # Adjust limit for more posts
    posts.append({
        "timestamp": datetime.fromtimestamp(submission.created_utc),
        "title": submission.title,
        "body": submission.selftext
    })

# Convert the collected posts into a DataFrame
df_reddit = pd.DataFrame(posts)
df_reddit.head()


# Combine title and body
df_reddit['message'] = df_reddit['title'] + " " + df_reddit['body']

# Simple text cleaning function
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\s+', ' ', text)       # remove extra spaces
    return text.strip()

df_reddit['cleaned_message'] = df_reddit['message'].apply(clean_text)

analyzer = SentimentIntensityAnalyzer()
df_reddit['sentiment_neg'] = df_reddit['cleaned_message'].apply(lambda text: analyzer.polarity_scores(text)['neg'])

def chunk_text(text, chunk_size=300):
    """
    Splits a string into chunks of chunk_size words each.
    Returns a generator that yields each chunk.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])


from transformers import pipeline

# ... (your existing code) ...

# Before using get_stress_score, initialize emotion_classifier:
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)


def get_stress_score(text):
    # Define stress-related emotions
    stress_emotions = ['sadness', 'anger', 'fear']

    # To collect stress scores for each chunk
    chunk_scores = []

    # Split the text into chunks (300 words per chunk, adjustable)
    for chunk in chunk_text(text, chunk_size=300):
        # Get emotion results for the chunk with truncation
        result = emotion_classifier(chunk, truncation=True, max_length=512, top_k=None)

        # Check if result is valid and has at least one element
        if result and isinstance(result, list) and len(result) > 0:
            # Expecting result[0] to be a list of dictionaries
            emotion_list = result[0]

            # If emotion_list is a dictionary instead of a list, wrap it in a list
            if isinstance(emotion_list, dict):
                emotion_list = [emotion_list]

            # Ensure emotion_list is a list before proceeding
            if isinstance(emotion_list, list):
                # Extract scores for stress-related emotions; add type check for each item
                scores = [d['score'] for d in emotion_list
                          if isinstance(d, dict) and d.get('label', '').lower() in stress_emotions]

                if scores:
                    chunk_scores.append(max(scores))
            else:
                print("Unexpected format in emotion_list:", emotion_list)

    # If no valid chunk scores were found, return 0
    if not chunk_scores:
        return 0

    # For this example, we return the maximum stress score across all chunks
    return max(chunk_scores)

# Apply the function to your DataFrame
df_reddit['stress_score'] = df_reddit['cleaned_message'].apply(get_stress_score)
print(df_reddit[['cleaned_message', 'stress_score']].head())

# ... (rest of your code) ...



def compute_burnout_score(text):
    # 1. Get negative sentiment from VADER
    neg_sentiment = analyzer.polarity_scores(text)['neg']

    # 2. Get stress score using your chunked emotion detection function
    stress = get_stress_score(text)

    # 3. Count burnout-related keywords
    burnout_keywords = ['exhausted', 'overwhelmed', 'burnt out', 'stressed', 'tired']
    keyword_score = sum(1 for kw in burnout_keywords if kw in text.lower())

    # 4. Combine the three metrics into one composite score
    # Adjust weights as needed (here: sentiment: 0.5, stress: 0.3, keyword: 0.2)
    composite_score = neg_sentiment * 0.5 + stress * 0.3 + keyword_score * 0.2
    return composite_score

# Now, add a new column 'burnout_score' to your DataFrame:
df_reddit['burnout_score'] = df_reddit['cleaned_message'].apply(compute_burnout_score)

# Check a sample of the results:
print(df_reddit[['cleaned_message', 'stress_score', 'burnout_score']].head())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(df_reddit['timestamp'], df_reddit['burnout_score'], marker='o', linestyle='-')
plt.xlabel("Timestamp")
plt.ylabel("Composite Burnout Score")
plt.title("Burnout Trends Over Time from Reddit Posts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Step 1: Create Binary Labels
# ----------------------------
# Use the median burnout score as the threshold for labeling
median_score = df_reddit['burnout_score'].median()
df_reddit['label'] = (df_reddit['burnout_score'] > median_score).astype(int)

# Check label distribution (should be roughly balanced)
print(df_reddit['label'].value_counts())

# ----------------------------
# Step 2: Split Data into Training and Validation Sets (Stratified)
# ----------------------------
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df_reddit,
    test_size=0.2,
    random_state=42,
    stratify=df_reddit['label']
)

# ----------------------------
# Step 3: Set Up the BERT Tokenizer and Model
# ----------------------------
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from torch.utils.data import Dataset

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("burnout_model")
tokenizer = BertTokenizer.from_pretrained("burnout_model")


# ----------------------------
# Step 4: Create a Custom Dataset Class
# ----------------------------
class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare the texts and labels lists from the DataFrame
train_texts = train_df['cleaned_message'].tolist()
train_labels = train_df['label'].tolist()
val_texts = val_df['cleaned_message'].tolist()
val_labels = val_df['label'].tolist()

train_dataset = RedditDataset(train_texts, train_labels, tokenizer, max_length=256)
val_dataset = RedditDataset(val_texts, val_labels, tokenizer, max_length=256)

# ----------------------------
# Step 5: Define Training Arguments and Metrics
# ----------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                      # Increase epochs to allow more training
    per_device_train_batch_size=4,           # Smaller batch size for more gradient updates
    per_device_eval_batch_size=4,
    warmup_steps=100,
    learning_rate=2e-5,                      # Lower learning rate can help with fine tuning
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",                   # Corrected argument name
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=[],                           # Disable wandb logging if not needed
    seed=42                                 # Ensure reproducibility
)

import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# ----------------------------
# Step 6: Initialize the Trainer and Fine-Tune BERT
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop early if no improvement
)

# Begin fine-tuning
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# 📦 Step 1: Install dependencies


# 📦 Step 2: Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import nest_asyncio
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import uvicorn

# 🧠 Step 3: Load model & tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dummy 3-class BERT model for demo (replace with your trained model path if saved)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.eval()

# 🔤 Step 4: Label encoder (manual simulation — make sure it matches your training)
le = LabelEncoder()
le.fit(["Low", "Medium", "High"])

# ⚙️ Step 5: Define FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_burnout(data: TextInput):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    prediction = torch.argmax(output.logits, dim=1).item()
    label = "High" if prediction == 1 else "Low"
    return {"burnout_level": label}
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use a specific domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)
# 🔗 Step 6: Run FastAPI with ngrok inside Colab
# (this must be the last cell so it doesn't block execution)

# Allow nested asyncio loop in Colab
nest_asyncio.apply()        

#Start ngrok tunnel
ngrok.set_auth_token("2zMzRlY7iQWyN7HqxraHiZmuzsV_5EPzPPaHtaCksHoZ2KaTv")  # 🔒 Replace with your token
public_url = ngrok.connect(8000)
print(f"🚀 Public URL: {public_url}")

# Run FastAPI app
uvicorn.run(app, host="127.0.0.1", port=8000);

print("🔥 Label distribution after balancing:")
print(df['label'].value_counts())