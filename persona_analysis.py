import pandas as pd
import nltk
import spacy
from nltk import FreqDist
from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import textstat

nltk.download('punkt')  # Download the punkt tokenizer
nltk.download('vader_lexicon')  # Download the VADER sentiment analysis model
nltk.download('averaged_perceptron_tagger')  # Download the POS tagger

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Replace 'path/to/your_file.csv' with the actual path to your CSV file
csv_file = 'message_history.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Filter rows where the sender is yourself
your_messages = df[df['is_from_me'] == 1]

# Extract the text_message column values
your_text_messages = your_messages['message'].tolist()

def tokenize_and_clean(text):
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalnum()]
    return words

def analyze_style(text_messages):
    all_words = []
    all_phrases = []
    all_pos_tags = []
    all_emojis = []
    sentence_lengths = []
    function_word_counts = []
    readability_scores = []

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []

    # Define a regex pattern for matching emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    # Function words list
    function_words = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "of", "to", "for", "with", "about", "as", "by", "like", "if", "so", "is", "am", "are", "was", "were", "be", "being", "been", "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]

    for message in text_messages:
        if not isinstance(message, str):
            print(message)
            continue
        # Tokenize the message into words
        words = tokenize_and_clean(message)

        # Add the words to the all_words list
        all_words.extend(words)

        # Generate phrases using n-grams (you can change n for different phrase lengths)
        phrases = list(ngrams(words, 2))  # Here we're using bigrams (2-word phrases)

        # Add the phrases to the all_phrases list
        all_phrases.extend(phrases)

        # Perform POS tagging
        pos_tags = nltk.pos_tag(words)
        all_pos_tags.extend(pos_tags)

        # Find and count emojis
        emojis = emoji_pattern.findall(message)
        all_emojis.extend(emojis)

        # Calculate sentence length
        sentence_lengths.append(len(words))

        # Calculate sentiment score
        sentiment_scores.append(sia.polarity_scores(message)['compound'])

        # Count function words
        function_word_count = sum([1 for word in words if word.lower() in function_words])
        function_word_counts.append(function_word_count)

        # Calculate readability score
        readability_scores.append(textstat.flesch_reading_ease(message))

    # Compute word frequency
    word_freq = FreqDist(all_words)

    # Compute phrase frequency
    phrase_freq = Counter(all_phrases)

    # Compute POS tag frequency
    pos_tag_freq = Counter([tag for word, tag in all_pos_tags])

    # Compute emoji frequency
    emoji_freq = Counter(all_emojis)

    # Calculate average sentence length
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

    # Calculate average sentiment score
    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

    # Calculate lexical diversity
    lexical_diversity = len(set(all_words)) / len(all_words)

    # Calculate average function word count
    avg_function_word_count = sum(function_word_counts) / len(function_word_counts)

    # Calculate average readability score
    avg_readability_score = sum(readability_scores) / len(readability_scores)

    return {
        "common_words": word_freq.most_common(5),
        "common_phrases": phrase_freq.most_common(5),
        "common_pos_tags": pos_tag_freq.most_common(5),
        "common_emojis": emoji_freq.most_common(5),
        "average_sentence_length": avg_sentence_length,
        "average_sentiment_score": avg_sentiment_score,
        "lexical_diversity": lexical_diversity,
        "average_function_word_count": avg_function_word_count,
        "average_readability_score": avg_readability_score
    }

analysis = analyze_style(your_text_messages)

# Extract the linguistic features from the analysis
common_words = ', '.join([word for word, freq in analysis['common_words']])
common_phrases = ', '.join([' '.join(phrase) for phrase, freq in analysis['common_phrases']])
common_pos_tags = ', '.join([tag for tag, freq in analysis['common_pos_tags']])
common_emojis = ', '.join([emoji for emoji, freq in analysis['common_emojis']])
avg_sentence_length = round(analysis['average_sentence_length'], 2)
avg_sentiment_score = round(analysis['average_sentiment_score'], 2)
lexical_diversity = round(analysis['lexical_diversity'], 2)
avg_function_word_count = round(analysis['average_function_word_count'], 2)
avg_readability_score = round(analysis['average_readability_score'], 2)

# Create a persona prompt
persona_prompt = f"The user often uses the following words: {common_words}, the following phrases: {common_phrases}, and the following POS tags: {common_pos_tags}. They frequently use these emojis: {common_emojis}. Their average sentence length is {avg_sentence_length} words, their average sentiment score is {avg_sentiment_score}, their lexical diversity is {lexical_diversity}, their average function word count is {avg_function_word_count}, and their average readability score is {avg_readability_score}. Please answer the following question in their style: What's your favorite hobby?"

print(persona_prompt)

