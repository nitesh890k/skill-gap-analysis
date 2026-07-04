from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

sentence = "I woke up early this morning and felt excited for the day."

scores = analyzer.polarity_scores(sentence)
print("Sentiment Scores:", scores)

compound = scores['compound']
if compound > 0:
    print("Overall Sentiment: Positive 😊")
elif compound < 0:
    print("Overall Sentiment: Negative 😞")
else:
    print("Overall Sentiment: Neutral 😐")
