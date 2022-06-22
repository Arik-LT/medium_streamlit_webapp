import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from textblob import TextBlob, Word
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import scipy

nltk.download('stopwords')

st.title('Article Clap Analysis')
st.subheader('Find out details about your medium article')


@st.cache(allow_output_mutation=True)
def load_data_for_model():
    df = pd.read_csv(
        'df_final_no_numbers.csv')

    df['subtitle'] = df['subtitle'].apply(lambda x: 0 if x == '-' else 1)
    df['subtitle'] = df['subtitle'].apply(str)
    df = df[(df['publication'] != 'personal-growth')
            & (df['publication'] != 'uxplanet')]
    df.reset_index(drop=True, inplace=True)

    return df


df = load_data_for_model()

st.subheader('User input Parameters')

col1, col2 = st.columns(2)

with col1:
    # Publication
    publication = st.selectbox(
        'In what publication will you publish your article?',
        ('towardsdatascience', 'swlh', 'writingcooperative',
         'datadriveninvestor', 'uxdesign', 'the-mission', 'startup-grind',
         'better-humans', 'better-marketing'))

with col2:
    # Followers
    followers = st.number_input('How many followers do you have?', step=1)

# title and text inputs
title = st.text_input('Write here the title for your medium article')
text = st.text_area('Write here the text for your medium article')


if publication == 'better-marketing':
    better_marketing = 1
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 0
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 0
if publication == 'startup-grind':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 1
    swlh = 0
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 0
if publication == 'towardsdatascience':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 0
    the_mission = 0
    towardsdatascience = 1
    uxdesign = 0
    writingcooperative = 0
if publication == 'swlh':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 1
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 0
if publication == 'writingcooperative':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 0
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 1
if publication == 'uxdesign':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 0
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 1
    writingcooperative = 0
if publication == 'datadriveninvestor':
    better_marketing = 0
    datadriveninvestor = 1
    startup_grind = 0
    swlh = 0
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 0
if publication == 'the-mission':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 0
    the_mission = 1
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 0
if publication == 'better-humans':
    better_marketing = 0
    datadriveninvestor = 0
    startup_grind = 0
    swlh = 0
    the_mission = 0
    towardsdatascience = 0
    uxdesign = 0
    writingcooperative = 0

col1, col2 = st.columns(2)

with col1:
    check_yes = st.checkbox('Does your article have a subtitle?')
    if check_yes:
        st.write('Great!')
        subtitle_1 = 1
    else:
        subtitle_1 = 0

with col2:
    # day of the week
    date = st.date_input('When will you publish your article?')
    day_of_week = date.today().weekday()


col1, col2, col3 = st.columns(3)

with col1:
    # length of text
    number_of_words = len(text.split())
    st.metric(label='Number of Words:', value=number_of_words)

sentiment = TextBlob(text).sentiment
polarity = sentiment[0]
subjectivity = sentiment[1]

with col2:

    st.metric(label='Polarity score: ',
              value=round(polarity, 2))

with col3:
    st.metric(label='Subjectivity score: ', value=round(subjectivity, 2))


if day_of_week == 0:
    day_of_the_week_1 = 0
    day_of_the_week_2 = 0
    day_of_the_week_3 = 0
    day_of_the_week_4 = 0
    day_of_the_week_5 = 0
    day_of_the_week_6 = 0
if day_of_week == 1:
    day_of_the_week_1 = 1
    day_of_the_week_2 = 0
    day_of_the_week_3 = 0
    day_of_the_week_4 = 0
    day_of_the_week_5 = 0
    day_of_the_week_6 = 0
if day_of_week == 2:
    day_of_the_week_1 = 0
    day_of_the_week_2 = 1
    day_of_the_week_3 = 0
    day_of_the_week_4 = 0
    day_of_the_week_5 = 0
    day_of_the_week_6 = 0
if day_of_week == 3:
    day_of_the_week_1 = 0
    day_of_the_week_2 = 0
    day_of_the_week_3 = 1
    day_of_the_week_4 = 0
    day_of_the_week_5 = 0
    day_of_the_week_6 = 0
if day_of_week == 4:
    day_of_the_week_1 = 0
    day_of_the_week_2 = 0
    day_of_the_week_3 = 0
    day_of_the_week_4 = 1
    day_of_the_week_5 = 0
    day_of_the_week_6 = 0
if day_of_week == 5:
    day_of_the_week_1 = 0
    day_of_the_week_2 = 0
    day_of_the_week_3 = 0
    day_of_the_week_4 = 0
    day_of_the_week_5 = 1
    day_of_the_week_6 = 0
if day_of_week == 6:
    day_of_the_week_1 = 0
    day_of_the_week_2 = 0
    day_of_the_week_3 = 0
    day_of_the_week_4 = 0
    day_of_the_week_5 = 0
    day_of_the_week_6 = 1


stop_words = stopwords.words('english')

data = {
    'title': title,
    'text': text,
    'author_followers': followers,
    'number_of_words': number_of_words,
    'polarity': polarity,
    'subjectivity': subjectivity,
    'publication_better-marketing': better_marketing,
    'publication_datadriveninvestor': datadriveninvestor,
    'publication_startup-grind': startup_grind,
    'publication_swlh': swlh,
    'publication_the-mission': the_mission,
    'publication_towardsdatascience': towardsdatascience,
    'publication_uxdesign': uxdesign,
    'publication_writingcooperative': writingcooperative,
    'day_of_the_week_1': day_of_the_week_1,
    'day_of_the_week_2': day_of_the_week_2,
    'day_of_the_week_3': day_of_the_week_3,
    'day_of_the_week_4': day_of_the_week_4,
    'day_of_the_week_5': day_of_the_week_5,
    'day_of_the_week_6': day_of_the_week_6,
    'subtitle_1': subtitle_1
}

user_df = pd.DataFrame(data, index=[0])


def clean_text(df, column_name):
    # lower case
    df[column_name] = df[column_name].apply(
        lambda x: " ".join(word.lower() for word in str(x).split()))

    # remove punctuation
    df[column_name] = df[column_name].str.replace('[^a-z ]', '')

    # this replaces two or more white spcaes with just one
    df[column_name] = df[column_name].str.replace(r"\s\s+", ' ')

    # remove stop words
    df[column_name] = df[column_name].apply(lambda x: " ".join(
        word for word in x.split() if word not in stop_words))

    # lemmatize
    df[column_name] = df[column_name].apply(
        lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

    return df


user_df = clean_text(user_df, 'text')
user_df = clean_text(user_df, 'title')


@st.experimental_memo
def xgb_model(df):
    #  Run Original Model
    X_ = df[['title', 'text', 'subtitle', 'author_followers', 'publication',
            'number_of_words', 'day_of_the_week', 'polarity', 'subjectivity']]
    X_ = pd.get_dummies(
        X_, columns=['publication', 'day_of_the_week', 'subtitle'], drop_first=True)
    y = df.claps_binary

    X_train, X_test, y_train, y_test = train_test_split(
        X_, y, test_size=0.33, random_state=42, stratify=y)

    # Training
    cvec = CountVectorizer(ngram_range=(1, 2), max_features=1000000)
    X_train_text = cvec.fit_transform(X_train.text)

    cvec_2 = CountVectorizer(ngram_range=(1, 2))
    X_train_title = cvec_2.fit_transform(X_train.title)

    train_to_sparse = X_train.drop(['title', 'text'], axis=1)
    sparse_df_train = scipy.sparse.csr_matrix(train_to_sparse.values)

    # Merge all three together
    X_train = scipy.sparse.hstack(
        (X_train_text, X_train_title, sparse_df_train))

    # Testing
    X_test_text = cvec.transform(X_test.text)
    X_test_title = cvec_2.transform(X_test.title)
    test_to_sparse = X_test.drop(['title', 'text'], axis=1)
    sparse_df_test = scipy.sparse.csr_matrix(test_to_sparse.values)

    # Merge all columns together
    X_test = scipy.sparse.hstack((X_test_text, X_test_title, sparse_df_test))

    # XGB Model
    xgb_model = XGBClassifier(use_label_encoder=False,
                              eval_metric='mlogloss', n_jobs=8)
    xgb_model.fit(X_train, y_train)

    return xgb_model, cvec, cvec_2


model, cvec, cvec_2 = xgb_model(df)

# Transform user input to CVEc and sparse matrix
X_text = cvec.transform(user_df.text)
X_title = cvec_2.transform(user_df.title)
to_sparse = user_df.drop(['title', 'text'], axis=1)
sparse_X = scipy.sparse.csr_matrix(to_sparse.values)
user_input = scipy.sparse.hstack((X_text, X_title, sparse_X))


proba = model.predict_proba(user_input)
st.metric(label='Probability of Belonging to Class One',
          value=f'{round(proba[0][1] * 100 ,2)}%')

st.subheader('Here are some similar articles based on your text!')


@st.cache(allow_output_mutation=True)
def load_data_for_tfidf():
    df = pd.read_csv(
        'data_fot_recommender.csv')

    # lower case
    df['text'] = df['text'].apply(lambda x: " ".join(
        word.lower() for word in str(x).split()))
    # remove punctuation
    df['text'] = df['text'].str.replace('[^a-z ]', '')
    # this replaces two or more white spcaes with just one
    df['text'] = df['text'].str.replace(r"\s\s+", ' ')
    # remove stop words
    df['text'] = df['text'].apply(lambda x: " ".join(
        word for word in x.split() if word not in stop_words))
    # lemmatize
    df['text'] = df['text'].apply(lambda x: " ".join(
        Word(word).lemmatize() for word in x.split()))

    return df


tfidf_df = load_data_for_tfidf()


with st.expander("See Article Suggestions"):
    st.write('Here we go:')
    # create a tf-idf vectorizer object
    tfidf = TfidfVectorizer(max_features=60000)
    X = tfidf.fit_transform(tfidf_df['text'])

    title2idx = pd.Series(tfidf_df.index, index=tfidf_df['title'])

    query = tfidf.transform(user_df.text)

    scores = cosine_similarity(query, X)
    scores = scores.flatten()

    recommended_idx = (-scores).argsort()[1:6]

    with st.container():
        for i, index in enumerate(recommended_idx):
            st.write(i+1, '-', tfidf_df['title'].iloc[index])
            st.write(tfidf_df['story_url'].iloc[index])
