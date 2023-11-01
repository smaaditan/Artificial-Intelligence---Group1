# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:23:58.285282Z","iopub.execute_input":"2022-04-01T10:23:58.2859Z","iopub.status.idle":"2022-04-01T10:24:09.553898Z","shell.execute_reply.started":"2022-04-01T10:23:58.285863Z","shell.execute_reply":"2022-04-01T10:24:09.552742Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:09.557946Z","iopub.execute_input":"2022-04-01T10:24:09.558285Z","iopub.status.idle":"2022-04-01T10:24:11.289719Z","shell.execute_reply.started":"2022-04-01T10:24:09.558239Z","shell.execute_reply":"2022-04-01T10:24:11.288628Z"}}
import nltk
import string
import pandas as pd
import nlp_utils as nu
import matplotlib.pyplot as plt
# Loading necessary libraries

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.291469Z","iopub.execute_input":"2022-04-01T10:24:11.29186Z","iopub.status.idle":"2022-04-01T10:24:11.324021Z","shell.execute_reply.started":"2022-04-01T10:24:11.291806Z","shell.execute_reply":"2022-04-01T10:24:11.323145Z"}}
f = open("D:\dataset.txt", "r")
print(f.read())
# reading the data 

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.325989Z","iopub.execute_input":"2022-04-01T10:24:11.326485Z","iopub.status.idle":"2022-04-01T10:24:11.348938Z","shell.execute_reply.started":"2022-04-01T10:24:11.32645Z","shell.execute_reply":"2022-04-01T10:24:11.348193Z"}}
df=pd.read_csv('D:\dataset.txt',names=('Query','Response'),sep=('\t'))
# Reading the data

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.350377Z","iopub.execute_input":"2022-04-01T10:24:11.350844Z","iopub.status.idle":"2022-04-01T10:24:11.373318Z","shell.execute_reply.started":"2022-04-01T10:24:11.350809Z","shell.execute_reply":"2022-04-01T10:24:11.37263Z"}}
df
# loading the data

# %% [markdown]
# ## Data Understanding

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.374476Z","iopub.execute_input":"2022-04-01T10:24:11.375122Z","iopub.status.idle":"2022-04-01T10:24:11.380857Z","shell.execute_reply.started":"2022-04-01T10:24:11.375088Z","shell.execute_reply":"2022-04-01T10:24:11.379797Z"}}
df.shape
# There are 3724 rows and 2 columns in our dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.3822Z","iopub.execute_input":"2022-04-01T10:24:11.382834Z","iopub.status.idle":"2022-04-01T10:24:11.398111Z","shell.execute_reply.started":"2022-04-01T10:24:11.382784Z","shell.execute_reply":"2022-04-01T10:24:11.397273Z"}}
df.columns
# Displaying the names of columns present in the dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.399544Z","iopub.execute_input":"2022-04-01T10:24:11.399939Z","iopub.status.idle":"2022-04-01T10:24:11.434222Z","shell.execute_reply.started":"2022-04-01T10:24:11.399901Z","shell.execute_reply":"2022-04-01T10:24:11.4335Z"}}
df.info()
# Checking information of the data

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.435391Z","iopub.execute_input":"2022-04-01T10:24:11.436024Z","iopub.status.idle":"2022-04-01T10:24:11.456867Z","shell.execute_reply.started":"2022-04-01T10:24:11.435985Z","shell.execute_reply":"2022-04-01T10:24:11.45614Z"}}
df.describe()
# Describe function shows us the frequency,unique and counts of all columns

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.460521Z","iopub.execute_input":"2022-04-01T10:24:11.460949Z","iopub.status.idle":"2022-04-01T10:24:11.471816Z","shell.execute_reply.started":"2022-04-01T10:24:11.46091Z","shell.execute_reply":"2022-04-01T10:24:11.470837Z"}}
df.nunique()
# nunique() function return number of unique elements in the object. 

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.473541Z","iopub.execute_input":"2022-04-01T10:24:11.474375Z","iopub.status.idle":"2022-04-01T10:24:11.486089Z","shell.execute_reply.started":"2022-04-01T10:24:11.474294Z","shell.execute_reply":"2022-04-01T10:24:11.485111Z"}}
df.isnull().sum()
# Checking for the presence of null values in the data. As we can see there are no null values present in the data

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.487583Z","iopub.execute_input":"2022-04-01T10:24:11.488528Z","iopub.status.idle":"2022-04-01T10:24:11.503Z","shell.execute_reply.started":"2022-04-01T10:24:11.488478Z","shell.execute_reply":"2022-04-01T10:24:11.501842Z"}}
df['Query'].value_counts()
# Checking the counts of the values present in the column 'Query'

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.504738Z","iopub.execute_input":"2022-04-01T10:24:11.505316Z","iopub.status.idle":"2022-04-01T10:24:11.517779Z","shell.execute_reply.started":"2022-04-01T10:24:11.505265Z","shell.execute_reply":"2022-04-01T10:24:11.516851Z"}}
df['Response'].value_counts()
# Checking the counts of the values present in the column 'Response'

# %% [markdown]
# ## Data Visualization

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.519468Z","iopub.execute_input":"2022-04-01T10:24:11.519988Z","iopub.status.idle":"2022-04-01T10:24:11.549116Z","shell.execute_reply.started":"2022-04-01T10:24:11.51994Z","shell.execute_reply":"2022-04-01T10:24:11.548191Z"}}
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.550719Z","iopub.execute_input":"2022-04-01T10:24:11.551221Z","iopub.status.idle":"2022-04-01T10:24:11.556655Z","shell.execute_reply.started":"2022-04-01T10:24:11.551174Z","shell.execute_reply":"2022-04-01T10:24:11.554514Z"}}
Text=df['Query']

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:11.558462Z","iopub.execute_input":"2022-04-01T10:24:11.558978Z","iopub.status.idle":"2022-04-01T10:24:15.050205Z","shell.execute_reply.started":"2022-04-01T10:24:11.558943Z","shell.execute_reply":"2022-04-01T10:24:15.049018Z"}}
sid = SentimentIntensityAnalyzer()
for sentence in Text:
     print(sentence)
        
     ss = sid.polarity_scores(sentence)
     for k in ss:
         print('{0}: {1}, ' .format(k, ss[k]), end='')
     print()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:15.052431Z","iopub.execute_input":"2022-04-01T10:24:15.052932Z","iopub.status.idle":"2022-04-01T10:24:16.701156Z","shell.execute_reply.started":"2022-04-01T10:24:15.052848Z","shell.execute_reply":"2022-04-01T10:24:16.700114Z"}}
analyzer = SentimentIntensityAnalyzer()
df['rating'] = Text.apply(analyzer.polarity_scores)
df=pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.Series)], axis=1)
### Creating a dataframe.

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:16.703392Z","iopub.execute_input":"2022-04-01T10:24:16.703958Z","iopub.status.idle":"2022-04-01T10:24:16.723936Z","shell.execute_reply.started":"2022-04-01T10:24:16.703904Z","shell.execute_reply":"2022-04-01T10:24:16.723278Z"}}
df

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:16.725114Z","iopub.execute_input":"2022-04-01T10:24:16.725383Z","iopub.status.idle":"2022-04-01T10:24:16.784585Z","shell.execute_reply.started":"2022-04-01T10:24:16.725331Z","shell.execute_reply":"2022-04-01T10:24:16.78371Z"}}
from wordcloud import WordCloud
# importing word cloud

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:16.785998Z","iopub.execute_input":"2022-04-01T10:24:16.786323Z","iopub.status.idle":"2022-04-01T10:24:16.794407Z","shell.execute_reply.started":"2022-04-01T10:24:16.786279Z","shell.execute_reply":"2022-04-01T10:24:16.793396Z"}}
def wordcloud(df, label):
    
    subset=df[df[label]==1]
    text=df.Query.values
    wc= WordCloud(background_color="black",max_words=1000)

    wc.generate(" ".join(text))

    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)
# visualising wordcloud    

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:16.795756Z","iopub.execute_input":"2022-04-01T10:24:16.796077Z","iopub.status.idle":"2022-04-01T10:24:17.747226Z","shell.execute_reply.started":"2022-04-01T10:24:16.796034Z","shell.execute_reply":"2022-04-01T10:24:17.746239Z"}}
wordcloud(df,'Query')
# top words in the query column

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:17.748618Z","iopub.execute_input":"2022-04-01T10:24:17.748908Z","iopub.status.idle":"2022-04-01T10:24:18.630181Z","shell.execute_reply.started":"2022-04-01T10:24:17.748871Z","shell.execute_reply":"2022-04-01T10:24:18.629546Z"}}
wordcloud(df,'Response')
# top words in the response column

# %% [markdown]
# ## Text-Normalization

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.6312Z","iopub.execute_input":"2022-04-01T10:24:18.631763Z","iopub.status.idle":"2022-04-01T10:24:18.635311Z","shell.execute_reply.started":"2022-04-01T10:24:18.631725Z","shell.execute_reply":"2022-04-01T10:24:18.634553Z"}}
# Removing special characters

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.636401Z","iopub.execute_input":"2022-04-01T10:24:18.636801Z","iopub.status.idle":"2022-04-01T10:24:18.646032Z","shell.execute_reply.started":"2022-04-01T10:24:18.636767Z","shell.execute_reply":"2022-04-01T10:24:18.645404Z"}}
import re
# importing regular expressions

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.647278Z","iopub.execute_input":"2022-04-01T10:24:18.647776Z","iopub.status.idle":"2022-04-01T10:24:18.657384Z","shell.execute_reply.started":"2022-04-01T10:24:18.647742Z","shell.execute_reply":"2022-04-01T10:24:18.656679Z"}}
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
# Lower case conversion

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.659044Z","iopub.execute_input":"2022-04-01T10:24:18.659567Z","iopub.status.idle":"2022-04-01T10:24:18.667549Z","shell.execute_reply.started":"2022-04-01T10:24:18.65952Z","shell.execute_reply":"2022-04-01T10:24:18.666792Z"}}
remove_n = lambda x: re.sub("\n", " ", x)
# removing \n and replacing them with empty value

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.668899Z","iopub.execute_input":"2022-04-01T10:24:18.66922Z","iopub.status.idle":"2022-04-01T10:24:18.679195Z","shell.execute_reply.started":"2022-04-01T10:24:18.669176Z","shell.execute_reply":"2022-04-01T10:24:18.678398Z"}}
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
# removing non ascii characters

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.68463Z","iopub.execute_input":"2022-04-01T10:24:18.685401Z","iopub.status.idle":"2022-04-01T10:24:18.690415Z","shell.execute_reply.started":"2022-04-01T10:24:18.685333Z","shell.execute_reply":"2022-04-01T10:24:18.689518Z"}}
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
# removing alpha numeric values

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.691607Z","iopub.execute_input":"2022-04-01T10:24:18.692203Z","iopub.status.idle":"2022-04-01T10:24:18.772851Z","shell.execute_reply.started":"2022-04-01T10:24:18.692169Z","shell.execute_reply":"2022-04-01T10:24:18.771839Z"}}
df['Query'] = df['Query'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
# using map function and applying the function on query column

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.774075Z","iopub.execute_input":"2022-04-01T10:24:18.774358Z","iopub.status.idle":"2022-04-01T10:24:18.848398Z","shell.execute_reply.started":"2022-04-01T10:24:18.774308Z","shell.execute_reply":"2022-04-01T10:24:18.847229Z"}}
df['Response'] = df['Response'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
# using map function and applying the function on response column

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.849673Z","iopub.execute_input":"2022-04-01T10:24:18.849918Z","iopub.status.idle":"2022-04-01T10:24:18.868641Z","shell.execute_reply.started":"2022-04-01T10:24:18.849886Z","shell.execute_reply":"2022-04-01T10:24:18.867705Z"}}
df
# final cleaned dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.870008Z","iopub.execute_input":"2022-04-01T10:24:18.870803Z","iopub.status.idle":"2022-04-01T10:24:18.878113Z","shell.execute_reply.started":"2022-04-01T10:24:18.870761Z","shell.execute_reply":"2022-04-01T10:24:18.877435Z"}}
pd.set_option('display.max_rows',3800)
# Displaying all rows in the dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:18.879806Z","iopub.execute_input":"2022-04-01T10:24:18.880459Z","iopub.status.idle":"2022-04-01T10:24:20.122798Z","shell.execute_reply.started":"2022-04-01T10:24:18.880412Z","shell.execute_reply":"2022-04-01T10:24:20.121985Z"}}
df

# %% [markdown]
# ### Important Sentence

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.124136Z","iopub.execute_input":"2022-04-01T10:24:20.125017Z","iopub.status.idle":"2022-04-01T10:24:20.130389Z","shell.execute_reply.started":"2022-04-01T10:24:20.124973Z","shell.execute_reply":"2022-04-01T10:24:20.129487Z"}}
imp_sent=df.sort_values(by='compound', ascending=False)
# arranging the compound column in descending order to find the best sentence. 

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.131746Z","iopub.execute_input":"2022-04-01T10:24:20.131999Z","iopub.status.idle":"2022-04-01T10:24:20.152835Z","shell.execute_reply.started":"2022-04-01T10:24:20.131959Z","shell.execute_reply":"2022-04-01T10:24:20.152077Z"}}
imp_sent.head(5)
# printing the first 5 rows

# %% [markdown]
# ### Top Positive Sentence

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.153923Z","iopub.execute_input":"2022-04-01T10:24:20.154265Z","iopub.status.idle":"2022-04-01T10:24:20.159423Z","shell.execute_reply.started":"2022-04-01T10:24:20.154234Z","shell.execute_reply":"2022-04-01T10:24:20.158681Z"}}
pos_sent=df.sort_values(by='pos', ascending=False)
# Arranging the dataframe by positive column in descending order to find the best postive sentence.

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.160592Z","iopub.execute_input":"2022-04-01T10:24:20.160976Z","iopub.status.idle":"2022-04-01T10:24:20.184585Z","shell.execute_reply.started":"2022-04-01T10:24:20.160944Z","shell.execute_reply":"2022-04-01T10:24:20.18383Z"}}
pos_sent.head(5)
# printing the first 5 rows

# %% [markdown]
# ### Top Negative Sentence

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.185646Z","iopub.execute_input":"2022-04-01T10:24:20.185983Z","iopub.status.idle":"2022-04-01T10:24:20.191541Z","shell.execute_reply.started":"2022-04-01T10:24:20.185953Z","shell.execute_reply":"2022-04-01T10:24:20.190432Z"}}
neg_sent=df.sort_values(by='neg', ascending=False)
# Arranging the dataframe by negative column in descending order to find the best negative sentence.

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.193108Z","iopub.execute_input":"2022-04-01T10:24:20.193469Z","iopub.status.idle":"2022-04-01T10:24:20.212455Z","shell.execute_reply.started":"2022-04-01T10:24:20.193423Z","shell.execute_reply":"2022-04-01T10:24:20.211503Z"}}
neg_sent.head(5)
# printing the first 5 rows

# %% [markdown]
# ### Top Neutral Sentence

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.214171Z","iopub.execute_input":"2022-04-01T10:24:20.21479Z","iopub.status.idle":"2022-04-01T10:24:20.225088Z","shell.execute_reply.started":"2022-04-01T10:24:20.214743Z","shell.execute_reply":"2022-04-01T10:24:20.224199Z"}}
neu_sent=df.sort_values(by='neu', ascending=False)
# Arranging the dataframe by negative column in descending order to find the best neutral sentence.

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.226466Z","iopub.execute_input":"2022-04-01T10:24:20.226727Z","iopub.status.idle":"2022-04-01T10:24:20.24609Z","shell.execute_reply.started":"2022-04-01T10:24:20.226684Z","shell.execute_reply":"2022-04-01T10:24:20.245327Z"}}
neu_sent.head(5)
# printing the first 5 rows

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.247091Z","iopub.execute_input":"2022-04-01T10:24:20.247963Z","iopub.status.idle":"2022-04-01T10:24:20.259146Z","shell.execute_reply.started":"2022-04-01T10:24:20.247925Z","shell.execute_reply":"2022-04-01T10:24:20.2581Z"}}
from sklearn.feature_extraction.text import TfidfVectorizer
# importing tfidf vectorizer

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.261184Z","iopub.execute_input":"2022-04-01T10:24:20.261599Z","iopub.status.idle":"2022-04-01T10:24:20.27065Z","shell.execute_reply.started":"2022-04-01T10:24:20.261561Z","shell.execute_reply":"2022-04-01T10:24:20.269877Z"}}
tfidf = TfidfVectorizer()
# Word Embedding - TF-IDF

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.272069Z","iopub.execute_input":"2022-04-01T10:24:20.272423Z","iopub.status.idle":"2022-04-01T10:24:20.401939Z","shell.execute_reply.started":"2022-04-01T10:24:20.272379Z","shell.execute_reply":"2022-04-01T10:24:20.400956Z"}}
factors = tfidf.fit_transform(df['Query']).toarray()
# changing column into array

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.403033Z","iopub.execute_input":"2022-04-01T10:24:20.403262Z","iopub.status.idle":"2022-04-01T10:24:20.434058Z","shell.execute_reply.started":"2022-04-01T10:24:20.403233Z","shell.execute_reply":"2022-04-01T10:24:20.433413Z"}}
tfidf.get_feature_names_out()
# displaying feature names

# %% [markdown]
# # Application

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.435297Z","iopub.execute_input":"2022-04-01T10:24:20.436159Z","iopub.status.idle":"2022-04-01T10:24:20.440998Z","shell.execute_reply.started":"2022-04-01T10:24:20.436109Z","shell.execute_reply":"2022-04-01T10:24:20.440211Z"}}
from sklearn.metrics.pairwise import cosine_distances
from nltk.stem import WordNetLemmatizer


# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:24:20.442108Z","iopub.execute_input":"2022-04-01T10:24:20.442938Z","iopub.status.idle":"2022-04-01T10:24:20.455063Z","shell.execute_reply.started":"2022-04-01T10:24:20.442889Z","shell.execute_reply":"2022-04-01T10:24:20.453964Z"}}
lemmatizer = WordNetLemmatizer()

query = 'who are you ?'
def chatbot(query):
    # step:-1 clean
    query = lemmatizer.lemmatize(query)
    # step:-2 word embedding - transform
    query_vector = tfidf.transform([query]).toarray()
    # step-3: cosine similarity
    similar_score = 1 -cosine_distances(factors,query_vector)
    index = similar_score.argmax() # take max index position
    # searching or matching question
    matching_question = df.loc[index]['Query']
    response = df.loc[index]['Response']
    pos_score = df.loc[index]['pos']
    neg_score = df.loc[index]['neg']
    neu_score = df.loc[index]['neu']
    confidence = similar_score[index][0]
    chat_dict = {'match':matching_question,
                'response':response,
                'score':confidence,
                'pos':pos_score,
                'neg':neg_score,
                'neu':neu_score}
    return chat_dict

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:27:08.020533Z","iopub.execute_input":"2022-04-01T10:27:08.020862Z","iopub.status.idle":"2022-04-01T10:27:08.091864Z","shell.execute_reply.started":"2022-04-01T10:27:08.020827Z","shell.execute_reply":"2022-04-01T10:27:08.090994Z"}}
#Sample
query = 'hi'
response = chatbot(query)
print(response)

# %% [markdown]
# **Run the below code in order to have the chatbot app, you should interact with that by running this code:**
# 

# %% [code] {"execution":{"iopub.status.busy":"2022-04-01T10:30:50.555622Z","iopub.execute_input":"2022-04-01T10:30:50.556249Z","iopub.status.idle":"2022-04-01T10:30:50.565707Z","shell.execute_reply.started":"2022-04-01T10:30:50.556202Z","shell.execute_reply":"2022-04-01T10:30:50.564465Z"}}
"""
while True:
    query = input('USER: ')
    query = 'hi'
    if query == 'exit':
        break
        
    response = chatbot(query)
    if response['score'] <= 0.2: # 
        print('BOT: Please rephrase your Question.')
    
    else:
        print('='*80)
        print('logs:\n Matched Question: %r\n Confidence Score: %0.2f \n PositiveScore: %r \n NegativeScore: %r\n NeutralScore: %r'%(
            response['match'],response['score']*100,response['pos'],response['neg'],response['neu']))
        print('='*80)
        print('BOT: ',response['response'])
        
"""

# %% [markdown]
# **Enjoy!**

# %% [code]
