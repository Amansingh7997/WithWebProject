import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# ----------------------------------
# 1. Load and Prepare Data from Reddit
# ----------------------------------
import praw
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re

# Reddit API setup
reddit = praw.Reddit(
    client_id='okx4v6PVXz86ilMbsJpC8Q',
    client_secret='bxSoHcTS4ci9XY58A9iLtGOrUDYy0g',
    user_agent='BurnoutDetectionApp by /u/Strange-Building-953'
)

# Fetch more posts
subreddit = reddit.subreddit("all")
query = "burnout work OR workplace"

posts = []
for submission in subreddit.search(query, limit=500):  # Increased to 500
    posts.append({
        "timestamp": datetime.fromtimestamp(submission.created_utc),
        "title": submission.title,
        "body": submission.selftext
    })

# Prepare dataframe
df = pd.DataFrame(posts)
df['message'] = df['title'] + " " + df['body']

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned_message'] = df['message'].apply(clean_text)

# Sentiment and stress
analyzer = SentimentIntensityAnalyzer()
df['sentiment_neg'] = df['cleaned_message'].apply(lambda text: analyzer.polarity_scores(text)['neg'])

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

stress_emotions = ['sadness', 'anger', 'fear']

def get_stress_score(text):
    result = emotion_classifier(text[:512])  # truncate
    if result and isinstance(result, list):
        emotion_list = result[0]
        scores = [d['score'] for d in emotion_list if d['label'].lower() in stress_emotions]
        return max(scores) if scores else 0
    return 0

df['stress_score'] = df['cleaned_message'].apply(get_stress_score)

# Composite burnout score
burnout_keywords = ['exhausted', 'overwhelmed', 'burnt out', 'stressed', 'tired']

def compute_burnout_score(text):
    neg = analyzer.polarity_scores(text)['neg']
    stress = get_stress_score(text)
    keyword_score = sum(1 for kw in burnout_keywords if kw in text.lower())
    return neg * 0.5 + stress * 0.3 + keyword_score * 0.2

df['burnout_score'] = df['cleaned_message'].apply(compute_burnout_score)

# Labels
median_score = df['burnout_score'].median()
df['label'] = (df['burnout_score'] > median_score).astype(int)

# 🔁 Balance the dataset manually
df_high = df[df['label'] == 1]
df_low = df[df['label'] == 0].sample(n=len(df_high), random_state=42)
df_balanced = pd.concat([df_high, df_low]).sample(frac=1).reset_index(drop=True)
df = df_balanced

print("🔥 Label distribution after balancing:")
print(df['label'].value_counts())

# ----------------------------------
# 2. Prepare Dataset for Training
# ----------------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_message'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BurnoutDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

train_dataset = BurnoutDataset(train_texts, train_labels)
val_dataset = BurnoutDataset(val_texts, val_labels)

# ----------------------------------
# 3. Train the Model
# ----------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./burnout_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# ----------------------------------
# 4. Save the Trained Model
# ----------------------------------
trainer.save_model("burnout_model")
tokenizer.save_pretrained("burnout_model")

print("✅ Model training complete and saved to 'burnout_model/'")
