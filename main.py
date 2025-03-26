import os
import nltk    

# set the download path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# add path to NLTK
nltk.data.path.append(nltk_data_path)

# download nltk corpus (first time only)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('vader_lexicon', download_dir=nltk_data_path)


# **Step 1** - Load dataset
import pandas as pd
# Load the amazon review dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')


# **Step 2** - Preprocess text
# Let’s create a function preprocess_text in which we first tokenize the documents using word_tokenize function from NLTK, 
# then we remove step words using stepwords module from NLTK and finally, we lemmatize the filtered_tokens using WordNetLemmatizer from NLTK.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# apply the function df
df['reviewText'] = df['reviewText'].apply(preprocess_text)


# **Step 3** - NLTK Sentiment Analyzer
# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# create get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment

# apply get_sentiment function
df['sentiment'] = df['reviewText'].apply(get_sentiment)

# The NLTK sentiment analyzer returns a score between -1 and +1. We have used a cut-off threshold of 0 in the get_sentiment 
# function above. Anything above 0 is classified as 1 (meaning positive). Since we have actual labels, we can evaluate 
# the performance of this method by building a confusion matrix.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(df['Positive'], df['sentiment']))

# We can also check the classification report:
from sklearn.metrics import classification_report
print(classification_report(df['Positive'], df['sentiment']))
