import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('./articles.csv')

# Ensure the dataframe contains the expected columns
expected_columns = ['headlines', 'description', 'content', 'url', 'category']
if not all(column in df.columns for column in expected_columns):
    raise ValueError(f"CSV file does not contain the required columns: {expected_columns}")

# Combine relevant text fields
df['combined_text'] = df['headlines'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['content'].fillna('')

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [word for word in text.split() if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

df['processed_text'] = df['combined_text'].apply(preprocess_text)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Function to fetch and preprocess article content from a URL
def fetch_and_preprocess_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('title').get_text() if soup.find('title') else ''
    paragraphs = soup.find_all('p')
    body = ' '.join([para.get_text() for para in paragraphs])
    combined_text = title + ' ' + body
    processed_text = preprocess_text(combined_text)
    return processed_text

# Console interface to classify an article based on its URL
def classify_article():
    url = input('Enter the URL of the article: ')
    processed_text = fetch_and_preprocess_article(url)
    features = vectorizer.transform([processed_text])
    prediction = classifier.predict(features)
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    print('Predicted Category:', predicted_category)

# Main loop to classify articles
if __name__ == '__main__':
    while True:
        classify_article()
        cont = input('Do you want to classify another article? (yes/no): ')
        if cont.lower() != 'yes':
            break
