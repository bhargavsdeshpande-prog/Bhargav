import json
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq_data.json", "r") as file:
    faqs = json.load(file)

# Text preprocessing
def preprocess(text):
    text = text.lower()

    for p in string.punctuation:
        text = text.replace(p, "")

    return text

# Get questions
questions = list(faqs.keys())

# Process questions
processed_questions = []

for q in questions:
    processed_questions.append(preprocess(q))

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# Chatbot function
def chatbot(user_input):

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarity.argmax()

    best_score = similarity[0][best_match_index]

    if best_score > 0.2:
        return faqs[questions[best_match_index]]

    return "Sorry, I don't understand."


if __name__ == "__main__":
    print(chatbot("What is Python?"))