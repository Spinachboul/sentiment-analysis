# import pandas as pd
# from transformers import BertTokenizer, TFBertForSequenceClassification
# import tensorflow as tf
#
# # Load BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
#
# # Function to perform sentiment analysis
# def perform_sentiment_analysis(platform, keyword, num_posts):
#     # Load dataset
#     data = pd.read_csv('sentimentdataset.csv')  # Replace 'your_dataset.csv' with the actual dataset file
#
#     # Filter dataset by platform and keyword
#     filtered_data = data[(data['Platform'] == platform) & (data['Text'].str.contains(keyword, case=False))]
#     filtered_data = filtered_data.head(num_posts)
#
#     if filtered_data.empty:
#         return "No posts found matching the criteria."
#
#     # Tokenize filtered data
#     encodings = tokenizer(filtered_data['Text'].tolist(), truncation=True, padding=True, max_length=128)
#
#     # Create dataset for prediction
#     test_dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(16)
#
#     # Predict sentiment
#     predictions = model.predict(test_dataset)
#     predicted_labels = tf.argmax(predictions.logits, axis=-1)
#
#     # Mapping predictions to sentiment labels
#     labels = ['Negative', 'Neutral', 'Positive']
#     result = []
#     for idx, label in enumerate(predicted_labels):
#         result.append(f"Post {idx + 1}: {filtered_data.iloc[idx]['Text'][:50]}... -> {labels[label]}")
#
#     return "\n".join(result)

import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode
model.eval()


def perform_sentiment_analysis(platform, keyword, num_posts):
    # Filter dataset based on platform and keyword
    filtered_df = df[(df['Platform'].str.lower() == platform.lower()) & (df['Text'].str.contains(keyword, case=False))]

    # Limit to the specified number of posts
    filtered_df = filtered_df.head(num_posts)

    if filtered_df.empty:
        return pd.DataFrame()

    sentiments = []

    for _, row in filtered_df.iterrows():
        text = row['Text']
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get the predicted class
        predicted_class_idx = torch.argmax(probabilities).item()

        # Map predicted class index to sentiment label
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        predicted_sentiment = sentiment_labels[predicted_class_idx]

        sentiments.append({
            'Text': text,
            'Predicted Sentiment': predicted_sentiment,
            'Original Sentiment': row['Sentiment'],
            'Platform': row['Platform']
        })

    # Return sentiment results as DataFrame
    return pd.DataFrame(sentiments)


