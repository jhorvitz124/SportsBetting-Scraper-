import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Function to scrape website and extract data
def scrape_website(url):
    # Make a GET request to the URL
    response = requests.get(url)

    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the main content (you may need to inspect the webpage HTML to find the right tags)
    main_content = soup.find('div', class_='main-content')

    # Get all the paragraphs
    paragraphs = main_content.find_all('p')

    # Extract text from paragraphs
    text = ' '.join([p.get_text() for p in paragraphs])

    return text

# Function to preprocess text using spaCy
def preprocess_text(text):
    doc = nlp(text)
    # Lemmatize and remove stopwords
    preprocessed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    return preprocessed_text

# Load labeled data for training the model
# Assume you have a CSV file with labeled data (1 for positive, 0 for negative)
data = pd.read_csv('labeled_data.csv')

# Preprocess the text
data['text'] = data['text'].apply(preprocess_text)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to get betting advice
def get_betting_advice(url):
    # Scrape website and extract data
    text = scrape_website(url)

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([preprocessed_text])

    # Make prediction using the trained classifier
    prediction = classifier.predict(text_vectorized)[0]

    if prediction == 1:
        betting_advice = "Betting advice: Favorable"
    else:
        betting_advice = "Betting advice: Not favorable"

    return betting_advice

# Example usage
url = "https://example.com/sports-betting-advice"
betting_advice = get_betting_advice(url)
print(betting_advice)
