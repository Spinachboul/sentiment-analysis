# from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, ConversationHandler
# import telegram.ext.filters as filters
# import logging
# import pandas as pd
# from transformers import BertTokenizer, TFBertForSequenceClassification
# import tensorflow as tf
# import sentiment
#
# # Initialize BERT model and tokenizer
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# # Enable logging
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Define states for conversation
# PLATFORM, KEYWORD, NUM_POSTS = range(3)
#
# # Bot token from BotFather
# TOKEN = "8024532085:AAGATHigY4LC_ai6pEzBeKgC_7taTAeWgnA"
#
# # Start command
# def start(update, context):
#     update.message.reply_text('Welcome! Use /analyze to perform sentiment analysis.')
#
# # Analyze command
# def analyze(update, context):
#     update.message.reply_text('Which platform would you like to analyze (e.g., Twitter)?')
#     return PLATFORM
#
# def platform(update, context):
#     context.user_data['platform'] = update.message.text
#     update.message.reply_text('Please enter the keyword for sentiment analysis.')
#     return KEYWORD
#
# def keyword(update, context):
#     context.user_data['keyword'] = update.message.text
#     update.message.reply_text('How many posts would you like to analyze?')
#     return NUM_POSTS
#
# def num_posts(update, context):
#     context.user_data['num_posts'] = int(update.message.text)
#
#     platform = context.user_data['platform']
#     keyword = context.user_data['keyword']
#     num_posts = context.user_data['num_posts']
#
#     update.message.reply_text(f'Analyzing {num_posts} posts on {platform} for keyword "{keyword}"...')
#
#     # Call sentiment analysis function here (Step 3)
#     result = sentiment.perform_sentiment_analysis(platform, keyword, num_posts)
#
#     update.message.reply_text(f'Results:\n{result}')
#     return ConversationHandler.END
#
# # Function to cancel the conversation
# def cancel(update, context):
#     update.message.reply_text('Operation cancelled.')
#     return ConversationHandler.END
#
# def main():
#     updater = Updater(TOKEN, use_context=True)
#     dp = updater.dispatcher
#
#     conv_handler = ConversationHandler(
#         entry_points=[CommandHandler('analyze', analyze)],
#         states={
#             PLATFORM: [MessageHandler(filters.TEXT, platform)],
#             KEYWORD: [MessageHandler(filters.TEXT, keyword)],
#             NUM_POSTS: [MessageHandler(filters.TEXT, num_posts)]
#         },
#         fallbacks=[CommandHandler('cancel', cancel)]
#     )
#
#     dp.add_handler(conv_handler)
#     dp.add_handler(CommandHandler('start', start))
#
#     updater.start_polling()
#     updater.idle()
#
# if __name__ == '_main_':
#     main()


from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, ConversationHandler
import telegram.ext.filters as filters
import logging
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Define states for conversation
PLATFORM, KEYWORD, NUM_POSTS = range(3)

# Bot token from BotFather
TOKEN = "YOUR_TOKEN_HERE"

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Initialize BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to preprocess text for BERT
def preprocess_text(text, tokenizer, max_length=128):
    tokens = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return tokens

# Sentiment analysis function using BERT
def perform_sentiment_analysis(platform, keyword, num_posts):
    # Filter dataset based on platform and keyword
    filtered_df = df[(df['Platform'] == platform) & (df['Text'].str.contains(keyword, case=False))]

    # Limit to the specified number of posts
    filtered_df = filtered_df.head(num_posts)

    sentiments = []

    for i, row in filtered_df.iterrows():
        # Preprocess text
        inputs = preprocess_text(row['Text'], tokenizer)

        # Predict sentiment using BERT
        outputs = model(inputs)
        prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]

        # Convert prediction to sentiment label
        sentiment_label = 'Positive' if prediction == 2 else 'Neutral' if prediction == 1 else 'Negative'
        sentiments.append({
            'Text': row['Text'],
            'Predicted Sentiment': sentiment_label,
            'Original Sentiment': row['Sentiment'],
            'Platform': row['Platform']
        })

    # Return sentiment results as DataFrame
    return pd.DataFrame(sentiments)

# Start command
def start(update, context):
    update.message.reply_text('Welcome! Use /analyze to perform sentiment analysis.')

# Analyze command to start the sentiment analysis process
def analyze(update, context):
    update.message.reply_text('Which platform would you like to analyze (e.g., Twitter)?')
    return PLATFORM

# Platform handler
def platform(update, context):
    context.user_data['platform'] = update.message.text
    update.message.reply_text('Please enter the keyword for sentiment analysis.')
    return KEYWORD

# Keyword handler
def keyword(update, context):
    context.user_data['keyword'] = update.message.text
    update.message.reply_text('How many posts would you like to analyze?')
    return NUM_POSTS

# Num_posts handler and perform sentiment analysis
def num_posts(update, context):
    context.user_data['num_posts'] = int(update.message.text)

    platform = context.user_data['platform']
    keyword = context.user_data['keyword']
    num_posts = context.user_data['num_posts']

    update.message.reply_text(f'Analyzing {num_posts} posts on {platform} for keyword "{keyword}"...')

    # Perform sentiment analysis
    result_df = perform_sentiment_analysis(platform, keyword, num_posts)

    # Convert result DataFrame to a string for bot output
    result_str = result_df.to_string(index=False)

    update.message.reply_text(f'Results:\n{result_str}')
    return ConversationHandler.END

# Cancel command
def cancel(update, context):
    update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END

# Main function to start the bot
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # Conversation handler for the sentiment analysis
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('analyze', analyze)],
        states={
            PLATFORM: [MessageHandler(filters.TEXT, platform)],
            KEYWORD: [MessageHandler(filters.TEXT, keyword)],
            NUM_POSTS: [MessageHandler(filters.TEXT, num_posts)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)
    dp.add_handler(CommandHandler('start', start))

    # Start the bot
    updater.start_polling()
    updater.idle()

# Run the bot
if __name__ == '__main__':
    main()
