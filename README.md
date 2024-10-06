# Sentiment Analysis Telegram Bot

This is a Telegram bot that performs sentiment analysis on social media posts using the VADER sentiment analysis model. The bot interacts with users through a conversation flow and allows them to choose a platform, enter a keyword, specify the number of posts, and then returns the sentiment analysis results.

## Features
- **Start Conversation**: `/start` to initiate interaction with the bot.
- **Platform Selection**: Choose which platform's data to analyze (e.g., Twitter, Instagram).
- **Keyword Search**: Enter a keyword for filtering posts.
- **Post Limit**: Specify the number of posts to analyze.
- **Sentiment Analysis**: The bot uses the VADER model to analyze the sentiment of the filtered posts and provides a summary of the analysis.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7+
- Telegram Bot API Token (see [Creating a Bot](#creating-a-telegram-bot))
- Required Python libraries (see [Installation](#installation))

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-repo/sentiment-analysis-bot.git
cd sentiment-analysis-bot
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate      # For Windows
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Download the VADER lexicon (if not downloaded automatically):

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Creating a Telegram Bot

1. Open Telegram and search for `BotFather`.
2. Start a chat with `BotFather` and create a new bot by typing `/newbot`.
3. Follow the instructions to name your bot and get the API token.
4. Copy your bot API token and save it for later use.

## Configuration

1. In the `main.py` file, replace the placeholder for the bot token with your actual Telegram bot token:

```python
TOKEN = "YOUR_BOT_API_TOKEN"
```

2. Place your dataset in the root directory and update the file path in the `main.py`:

```python
df = pd.read_csv('path_to_your_dataset.csv')
```

## Usage

1. Run the bot:

```bash
python main.py
```

2. Open Telegram and start a conversation with your bot using the `/start` command.

3. Follow the conversation flow:

    - The bot will ask you to select a platform (e.g., Twitter).
    - Provide a keyword to search for posts related to your topic.
    - Specify the number of posts you want to analyze.
    - The bot will return a list of posts along with the sentiment score for each.

### Command Reference

- **/start**: Starts the bot and initiates the conversation.
- **/analyze**: Starts the sentiment analysis process.
- **/cancel**: Cancels the current operation.

## Project Structure

```
.
├── main.py                 # Main script for the Telegram bot
├── requirements.txt        # Dependencies for the project
├── README.md               # Project documentation
└── path_to_your_dataset.csv # Your dataset file (replace this with actual path)
```

## Error Handling

If the bot encounters any errors, they will be logged in the console, and users will receive a message indicating what went wrong. For instance, if a user enters invalid input (e.g., non-integer for the number of posts), the bot will ask them to correct the input.

## Dependencies

The project depends on the following Python libraries:

- `python-telegram-bot`
- `pandas`
- `nltk`

You can install these dependencies using `pip` from the `requirements.txt` file.

## Example Usage

Here is an example of how the conversation flow works:

1. **User**: `/start`
2. **Bot**: `Welcome! Use /analyze to perform sentiment analysis.`
3. **User**: `/analyze`
4. **Bot**: `Which platform would you like to analyze (e.g., Twitter, Instagram)?`
5. **User**: `Twitter`
6. **Bot**: `Please enter the keyword for sentiment analysis.`
7. **User**: `AI`
8. **Bot**: `How many posts would you like to analyze?`
9. **User**: `5`
10. **Bot**: `Analyzing 5 posts on Twitter for keyword "AI"...`

   (Bot will then return sentiment analysis results.)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Notes

- Make sure you have the correct dataset format with relevant columns like `Platform`, `Text`, etc., as expected by the bot.
- Modify the dataset path and format based on your specific use case.

Let me know if you need further adjustments or any other details!
