import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)                 
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)        
    text = text.lower()                                
    text = re.sub(r'\s+', ' ', text).strip()          
    return text

def get_ngrams(text, n):
    tokens = text.split()
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return ngrams

def jaccard_similarity(set1, set2):
    return float(len(set1.intersection(set2))) / len(set1.union(set2)) if set1.union(set2) else 0.0


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


text1 = read_txt("example.txt")
text2 = read_txt("example2.txt")  


cleaned_text1 = clean_text(text1)
cleaned_text2 = clean_text(text2)


tokens1 = word_tokenize(cleaned_text1)
tokens2 = word_tokenize(cleaned_text2)


stop_words = set(stopwords.words('english'))
tokens1 = [w for w in tokens1 if w not in stop_words]
tokens2 = [w for w in tokens2 if w not in stop_words]


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokens1 = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens1]
tokens2 = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens2]


text1_for_vector = " ".join(tokens1)
text2_for_vector = " ".join(tokens2)


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([text1_for_vector, text2_for_vector])
similarity = cosine_similarity(vectors)[0][1]
plagiarism_percentage = similarity * 100


common_tokens = set(tokens1).intersection(set(tokens2))


bigrams1 = get_ngrams(text1_for_vector, 2)
bigrams2 = get_ngrams(text2_for_vector, 2)
trigrams1 = get_ngrams(text1_for_vector, 3)
trigrams2 = get_ngrams(text2_for_vector, 3)

bigram_jaccard = jaccard_similarity(bigrams1, bigrams2)
trigram_jaccard = jaccard_similarity(trigrams1, trigrams2)


print("\n--- Plagiarism Detection Report ---")
print(f"Similarity Score: {similarity:.4f}")
print(f"Plagiarism Percentage: {plagiarism_percentage:.2f} %")

print("\nCleaned Text 1:")
print(cleaned_text1)
print("Tokens 1:", tokens1)

print("\nCleaned Text 2:")
print(cleaned_text2)
print("Tokens 2:", tokens2)

print("\nCommon Words Between the Two Texts:")
print(common_tokens)

print("\nBigrams Jaccard Similarity:", f"{bigram_jaccard:.2f}")
print("Trigrams Jaccard Similarity:", f"{trigram_jaccard:.2f}")
print("Common Bigrams:", bigrams1.intersection(bigrams2))
print("Common Trigrams:", trigrams1.intersection(trigrams2))
