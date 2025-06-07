# Step 1: Import Libraries
import nltk
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot("ChatBot")
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("ChatBot: Bye!")
        break
    response = chatbot.get_response(user_input)
    print("ChatBot:", response)


# Step 2: Download Required NLTK Data
nltk.download('punkt_tab')         # ✅ This was missing!
nltk.download('wordnet')
nltk.download('omw-1.4')

# Step 3: Sample Chat Corpus
chat_corpus = """
Hi
Hello
How are you?
I am fine, thank you.
What is your name?
My name is ChatBot.
What can you do?
I can chat with you and answer simple questions.
Who created you?
I was created by a developer using Python.
Goodbye
Bye! Take care.
"""

# Step 4: Preprocess the Text
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Use line-by-line tokens instead of sentence splitting
sentence_tokens = chat_corpus.lower().strip().splitlines()
sentence_tokens = [line.strip() for line in sentence_tokens if line.strip() != ""]

def LemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Step 5: Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(tokenizer=Normalize)

# Step 6: Response Generator
def generate_response(user_input):
    temp_tokens = sentence_tokens + [user_input]
    tfidf = vectorizer.fit_transform(temp_tokens)
    similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]
    if score == 0:
        return "I'm sorry, I don't understand that."
    else:
        return sentence_tokens[idx]

# Step 7: Run the Chat Loop
print("ChatBot: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ").lower()
    if user_input == 'bye':
        print("ChatBot: Bye! Have a great day!")
        break
    else:
        print("ChatBot:", generate_response(user_input))
