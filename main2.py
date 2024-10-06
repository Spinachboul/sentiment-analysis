from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import logging
import nltk

nltk.download('vader_lexicon')

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define states for conversation
PLATFORM, KEYWORD, NUM_POSTS = range(3)

# Load and clean dataset
def load_and_clean_dataset():
    df = pd.read_csv('sentimentdataset.csv')
    # Clean whitespace from relevant columns
    df['Platform'] = df['Platform'].str.strip()
    df['Text'] = df['Text'].str.strip()
    return df

# Load your dataset
df = load_and_clean_dataset()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Welcome! Use /analyze to perform sentiment analysis.')

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Get unique platforms from the dataset
    platforms = df['Platform'].unique()
    platform_list = ', '.join(platforms)
    await update.message.reply_text(f'Available platforms: {platform_list}\nWhich platform would you like to analyze?')
    return PLATFORM

async def platform(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_platform = update.message.text.strip()

    if not df['Platform'].str.lower().isin([user_platform.lower()]).any():
        await update.message.reply_text(f'Platform "{user_platform}" not found. Please choose from available platforms.')
        return PLATFORM

    context.user_data['platform'] = user_platform
    await update.message.reply_text('Please enter the keyword for sentiment analysis.')
    return KEYWORD

async def keyword(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['keyword'] = update.message.text.strip()
    await update.message.reply_text('How many posts would you like to analyze?')
    return NUM_POSTS

async def num_posts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        num_posts = int(update.message.text)
        context.user_data['num_posts'] = num_posts

        platform = context.user_data['platform']
        keyword = context.user_data['keyword']

        await update.message.reply_text(f'Analyzing {num_posts} posts on {platform} for keyword "{keyword}"...')

        # Debug logging
        logger.info(f"Searching for platform: '{platform}'")
        logger.info(f"Available platforms in dataset: {df['Platform'].unique().tolist()}")

        platform_mask = df['Platform'].str.lower() == platform.lower()
        keyword_mask = df['Text'].str.lower().str.contains(keyword.lower())
        filtered_df = df[platform_mask & keyword_mask]

        logger.info(f"Found {len(filtered_df)} matching posts")

        if filtered_df.empty:
            await update.message.reply_text(
                f'No posts found matching your criteria.\n'
                f'Platform: {platform}\n'
                f'Keyword: {keyword}\n'
                f'Please try different parameters.'
            )
            return ConversationHandler.END

        filtered_df = filtered_df.head(num_posts)

        filtered_df['Sentiment_Score'] = filtered_df['Text'].apply(lambda text: sia.polarity_scores(text)['compound'])

        results = []
        for _, row in filtered_df.iterrows():
            sentiment_label = "Positive" if row['Sentiment_Score'] > 0 else "Negative" if row['Sentiment_Score'] < 0 else "Neutral"
            truncated_text = row['Text'][:100] + ('...' if len(row['Text']) > 100 else '')
            results.append(f"Post: {truncated_text}\nSentiment: {sentiment_label} ({row['Sentiment_Score']:.2f})\n")

        # Send results in chunks to avoid message length limits
        chunk_size = 3
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i + chunk_size]
            await update.message.reply_text('\n'.join(chunk))

        avg_sentiment = filtered_df['Sentiment_Score'].mean()
        sentiment_distribution = filtered_df['Sentiment_Score'].apply(
            lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
        ).value_counts()

        summary = (
            f'Analysis Complete!\n'
            f'Total Posts Analyzed: {len(filtered_df)}\n'
            f'Average Sentiment Score: {avg_sentiment:.2f}\n'
            f'Distribution:\n'
        )
        for sentiment, count in sentiment_distribution.items():
            summary += f'{sentiment}: {count} posts\n'

        await update.message.reply_text(summary)

    except ValueError:
        await update.message.reply_text('Please enter a valid number.')
        return NUM_POSTS
    except Exception as e:
        logger.error(f"Error in num_posts: {str(e)}", exc_info=True)
        await update.message.reply_text('An error occurred while processing your request. Please try again.')

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("An error occurred while processing your request. Please try again.")

def main():
    TOKEN = "API_KEY"
    application = ApplicationBuilder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('analyze', analyze)],
        states={
            PLATFORM: [MessageHandler(filters.TEXT & ~filters.COMMAND, platform)],
            KEYWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, keyword)],
            NUM_POSTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, num_posts)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('start', start))
    application.add_error_handler(error_handler)

    application.run_polling()

if __name__ == '__main__':
    main()