'''
@Author: Ayesha Siddika Nipu
'''

import os
import re
import time
import string
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score

import warnings   
warnings.filterwarnings('ignore')

colReview = 'review'
colSentiment = 'sentiment'
filename = 'IMDB Dataset.csv'

#Calculate elapsed time
def ConvertAndShowElapsedTime(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    print('Elapsed Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(h, m, s)) # Python 3

#Remove stopwords
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
    return text

# Remove url  
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Remove punct
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Remove html 
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Remove @username
def remove_username(text):
    return re.sub('@[^\s]+','',text)

# Remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Decontraction text
def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)
    
    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    return text 

# Seperate alphanumeric
def seperate_alphanumeric(text):
    words = text
    words = re.findall(r"[^\W\d_]+|\d+", words)
    return " ".join(words)

def cont_rep_char(text):
    tchr = text.group(0) 
    
    if len(tchr) > 1:
        return tchr[0:2] 

def unique_char(rep, text):
    substitute = re.sub(r'(\w)\1+', rep, text)
    return substitute

def char(text):
    substitute = re.sub(r'[^a-zA-Z]',' ',text)
    return substitute

def remove_tag(text):
    return text.replace("<br />", " ")

# convert Sentiments to 0=negative,1 = positive
def convert_Sentiment(sentiment):
    if  sentiment == "positive":
        return 1
    elif sentiment == "negative":
        return 0

#performs cleanup
def DataPreprocessing(df):
    df[colReview] = df[colReview].apply(lambda x : remove_username(x))
    df[colReview] = df[colReview].apply(lambda x : remove_tag(x))
    df[colReview] = df[colReview].apply(lambda x : remove_url(x))
    df[colReview] = df[colReview].apply(lambda x : remove_emoji(x))
    df[colReview] = df[colReview].apply(lambda x : decontraction(x))
    df[colReview] = df[colReview].apply(lambda x : seperate_alphanumeric(x))
    df[colReview] = df[colReview].apply(lambda x : unique_char(cont_rep_char,x))
    df[colReview] = df[colReview].apply(lambda x : char(x))
    df[colReview] = df[colReview].apply(lambda x : x.lower())
    df[colReview] = df[colReview].apply(lambda x : remove_stopwords(x))
    df[colSentiment] = df[colSentiment].apply(lambda x : convert_Sentiment(x))

def ImplementSVMUsingTfidf():
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_prediction =  svm.predict(X_test)
    #print(type(svm_prediction))
    result_acc = round(accuracy_score(svm_prediction,y_test), 2)
    print("The accuracy is: ", round(result_acc*100, 2), '% \t [SVM-TfIdf]')
    return svm_prediction

def ImplementSVMUsingCountVectorizer():
    svm_cv = SVC()
    svm_cv.fit(X_train_cv, y_train_cv)
    svm_prediction_cv =  svm_cv.predict(X_test_cv)
    result_acc_cv = round(accuracy_score(svm_prediction_cv, y_test_cv), 2)
    print("The accuracy is: ", round(result_acc_cv*100, 2), '% \t [SVM-CountVectorizer]')
    return svm_prediction_cv
    
def ImplementRF():
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    rf_prediction =  rf.predict(X_test)
    result_acc = accuracy_score(rf_prediction, y_test)
    print("The accuracy is: ", round(result_acc*100, 2), '% \t [RandomForest]')
    return rf_prediction
    
def ImplementGradientBoostingClassifier():
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train,y_train)
    gbc_prediction =  gbc.predict(X_test)
    result_acc = accuracy_score(gbc_prediction,y_test)
    print("The accuracy is: ", round(result_acc*100, 2), '% \t [GradientBoostingClassifier]')
    return gbc_prediction
    
def ImplementNaiveBayes():
    nb = MultinomialNB()
    nb.fit(X_train,y_train)
    nb_prediction =  nb.predict(X_test)
    result_acc = accuracy_score(nb_prediction,y_test)
    print("The accuracy is: ", round(result_acc*100, 2), '% \t [NaiveBayes]')
    return nb_prediction
    
def ImplementDecisionTree():
    ds = DecisionTreeClassifier()
    ds.fit(X_train,y_train)
    ds_prediction =  ds.predict(X_test)
    result_acc = accuracy_score(ds_prediction,y_test)
    print("The accuracy is: ", round(result_acc*100, 2), '% \t [DecisionTree]')
    return ds_prediction
    
def SplitTestTrainDatasetUsingTfIdf(df):
    X = df[colReview]
    y = df[colSentiment]
    # Apply TFIDF on cleaned data
    tfid = TfidfVectorizer()
    X_final =  tfid.fit_transform(X)
    # Split Data into train & test 
    X_train , X_test , y_train , y_test = train_test_split(X_final, y , test_size=0.2)
    
    return X_train , X_test , y_train , y_test

def SplitTestTrainDatasetUsingCountVectorizer(df):
    X = df[colReview]
    y = df[colSentiment]
    # Apply TFIDF on cleaned data
    cv = CountVectorizer()
    X_final =  cv.fit_transform(X)
    # Split Data into train & test 
    X_train , X_test , y_train , y_test = train_test_split(X_final, y , test_size=0.2)
    
    return X_train , X_test , y_train , y_test

def PlotClassificationReportSvmTfidf():
    #Classification Report
    cr = classification_report(y_test, svm_prediction)
    print("\nClassification Report: SVM Prediction - TfIdf\n------------------------------------------------------\n", cr)
    cm = confusion_matrix(y_test,svm_prediction)
    # plot confusion matrix 
    plt.figure(figsize=(6,4))
    sentiment_classes = ['Negative', 'Positive']
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix: SVM Tfidf', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    plt.show()
    
def PlotClassificationReportSvmCv():
    #Classification Report
    cr = classification_report(y_test_cv, svm_prediction_cv)
    print("\nClassification Report: SVM Prediction - CountVectorizer\n------------------------------------------------------\n", cr)
    cm = confusion_matrix(y_test_cv,svm_prediction_cv)
    # plot confusion matrix 
    plt.figure(figsize=(6,4))
    sentiment_classes = ['Negative', 'Positive']
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix: SVM CountVectorizer', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    plt.show()
    
def PlotClassificationReportNb():
    #Classification Report
    cr = classification_report(y_test, nb_prediction)
    print("\nClassification Report: Naive Bayes Prediction\n------------------------------------------------------\n", cr)
    cm = confusion_matrix(y_test,nb_prediction)
    # plot confusion matrix 
    plt.figure(figsize=(6,4))
    sentiment_classes = ['Negative', 'Positive']
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix: NaiveBayes', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    plt.show()
    

def PlotClassificationReportRandomForest():
    #Classification Report
    cr = classification_report(y_test, rf_prediction)
    print("\nClassification Report: Random Forest Prediction\n------------------------------------------------------\n", cr)
    cm = confusion_matrix(y_test,rf_prediction)
    # plot confusion matrix 
    plt.figure(figsize=(6,4))
    sentiment_classes = ['Negative', 'Positive']
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix: RandomForest', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    plt.show()
    
def PlotClassificationReportDecisionTree():
    #Classification Report
    cr = classification_report(y_test, dt_prediction)
    print("\nClassification Report: Decision Tree Prediction\n------------------------------------------------------\n", cr)
    cm = confusion_matrix(y_test, dt_prediction)
    # plot confusion matrix 
    plt.figure(figsize=(6,4))
    sentiment_classes = ['Negative', 'Positive']
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix: DecisionTree', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    plt.show()
    
def PlotClassificationReportGbcPrediction():    
    #Classification Report
    cr = classification_report(y_test, gbc_prediction)
    print("\nClassification Report: GBC Prediction\n------------------------------------------------------\n", cr)
    cm = confusion_matrix(y_test, gbc_prediction)
    # plot confusion matrix 
    plt.figure(figsize=(6,4))
    sentiment_classes = ['Negative', 'Positive']
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix: GBC', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    plt.show()
    
    
def PlotStatistics():
    PlotClassificationReportSvmTfidf()
    PlotClassificationReportSvmCv()
    PlotClassificationReportNb()
    PlotClassificationReportRandomForest()
    PlotClassificationReportDecisionTree()
    PlotClassificationReportGbcPrediction()
    
    
start = time.time()
df = pd.read_csv(os.getcwd() + "\\Movie\\" + filename)
DataPreprocessing(df)
X_train, X_test, y_train, y_test = SplitTestTrainDatasetUsingTfIdf(df)
X_train_cv, X_test_cv , y_train_cv , y_test_cv = SplitTestTrainDatasetUsingCountVectorizer(df)


svm_prediction = ImplementSVMUsingTfidf()
svm_prediction_cv = ImplementSVMUsingCountVectorizer()
nb_prediction = ImplementNaiveBayes()
rf_prediction = ImplementRF()
dt_prediction = ImplementDecisionTree()
gbc_prediction = ImplementGradientBoostingClassifier()

PlotStatistics()

end = time.time()
ConvertAndShowElapsedTime(end - start)
