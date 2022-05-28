from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus=[]
with open('reference.txt') as f1:
    corpus.append(f1.read())
    f1.close()
with open('test.txt') as f1:
    corpus.append(f1.read())
    f1.close()
vectorizer=TfidfVectorizer()
trsfm=vectorizer.fit_transform(corpus)    
score=cosine_similarity(trsfm[0],trsfm)[0][1]*100
print(score)