import encodings
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
# from keybert import KeyBERT
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import language_tool_python
from rake_nltk import Rake
tool = language_tool_python.LanguageTool('en-US')
rake_nltk_var = Rake();
# kw_model = KeyBERT(model='all-mpnet-base-v2')
def calculate_cosine_similarity(corpus):
    vectorizer=TfidfVectorizer()
    trsfm=vectorizer.fit_transform(corpus)    
    score=cosine_similarity(trsfm[0],trsfm)[0][1]*10
    return round(score,2)

def stemmer(keywords_list):
    ps = PorterStemmer()
    for i in range(len(keywords_list)):
        keywords_list[i] = ps.stem(keywords_list[i])
    return keywords_list

def lemmatize(keywords_list):
    lemmatizer = WordNetLemmatizer()
    for i in range(len(keywords_list)):
        keywords_list[i] = lemmatizer.lemmatize(keywords_list[i])
    return keywords_list
corpus=[]
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/success', methods=['POST','GET'])
def success():
    if request.method=='POST':
        f=None
        f=request.files['file']
        if f == None:
            return render_template('erroredirect.html',message='empty_file')
        fname=f.filename
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return render_template('erroredirect.html',message='image_file')
        answer=f.read().decode('utf-8')
        matches = tool.check(answer)
        keywords_correct_answer_list = None
        # keywords_answer = kw_model.extract_keywords(answer, 
        #                             keyphrase_ngram_range=(0,1), 
        #                             stop_words='english', 
        #                             highlight=False,
        #                             )

        # keywords_answer_list= list(dict(keywords_answer).keys())
        rake_nltk_var.extract_keywords_from_text(answer)
        keywords_answer_list = rake_nltk_var.get_ranked_phrases()
        f.close()
        # f.save(f.filename)
        with open('reference.txt',encoding='utf-8') as fgt:
            corpus.append(fgt.read())
            correct_answer=corpus[0]
            # keywords_correct_answer = kw_model.extract_keywords(correct_answer,
            #                                                     keyphrase_ngram_range=(0,1),
            #                                                     stop_words='english',
            #                                                     highlight=False,
            #                                                     )
            
            # keywords_correct_answer_list = list(dict(keywords_correct_answer).keys())
            rake_nltk_var.extract_keywords_from_text(correct_answer)
            keywords_correct_answer_list = rake_nltk_var.get_ranked_phrases()
            fgt.close()
        
        common_keywords = 0
        keywords_answer_list = stemmer(keywords_answer_list)
        keywords_correct_answer_list = stemmer(keywords_correct_answer_list)
        keywords_answer_list = lemmatize(keywords_answer_list)
        keywords_correct_answer_list = lemmatize(keywords_correct_answer_list)
        keywords_answer_list_set = set(keywords_answer_list)
        keywords_correct_answer_list_set = set(keywords_correct_answer_list)
        print(keywords_answer_list)
        print(keywords_correct_answer_list)
        for ka in keywords_answer_list_set:
            for kca in keywords_correct_answer_list_set:
                if ka == kca:
                    common_keywords+=1
        complete_list = keywords_answer_list + keywords_correct_answer_list
        unique_keywords = len(np.unique(complete_list))
        keywords_match_score = (common_keywords/unique_keywords)*10
        corpus.append(answer)
        cosine_sim_score = calculate_cosine_similarity(corpus)
        score=((6/10)*(cosine_sim_score))+((4/10)*(keywords_match_score))
        if score >= 10:
            score = 10
        corpus.clear()
        if len(matches)>0:
            score = score - len(matches)
        if score<0:
            score = 0
        print("Errors\t",len(matches))
        print('Cosine_sim_score:\t',cosine_sim_score)
        print('keyword_match_score:\t',keywords_match_score)
        return render_template('success.html',name=fname,answer=answer,score=score,correct_answer=correct_answer,matches=len(matches))  
    if request.method=='GET':
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
    