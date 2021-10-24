import sys,os
sys.path.append(os.path.realpath('..'))
filename = 'r-cryptocurrency/2017-11.csv'
import pandas as pd
df = pd.read_csv(filename, encoding='utf-8')
#df.head
df.columns
row_headers = ['post_id','useraccount_id', 'thread_id', 'post/comment text', 'timestamp', 'karma points awarded', 'post type', 'role']
df = pd.read_csv(filename, names=row_headers, skiprows=1)
reordered_columns = ['timestamp',  'post_id', 'useraccount_id', 'thread_id', 'post type', 'role', 'karma points awarded', 'post/comment text']
df = df[reordered_columns]
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, errors='coerce')
df = df.sort_values(by='timestamp')
df.to_csv('sna_preprocessed.csv', index=False, header=True)
print(df.head())
print (df.columns)
df.shape
print(df['post_id'].nunique())
print(df['thread_id'].nunique())
print(df['useraccount_id'].nunique())
print(df['role'].nunique())
print(df['karma points awarded'].nunique())
print(df['post type'].nunique())

print(df['role'].unique())
print(df['karma points awarded'].unique())
print(df['post type'].unique())
df_users = df.groupby('useraccount_id')['post type'].value_counts().unstack().fillna(0)
#df_users.to_csv('sna_processed_data-users.csv', header=True)
df_users.describe()
df_threads = df.groupby('thread_id')['post type'].value_counts().unstack().fillna(0)
#df_threads.to_csv('sna_processed_data-threads.csv', header=True)
print(df_threads.head())
df_threads.describe()
df_threads_users = df.groupby('thread_id')['useraccount_id'].value_counts().unstack().fillna(0)
df_threads_users.to_csv('sna_processed_data-threads-users.csv', header=True)
print(df_threads_users.head())
df_threads_roles = df.groupby('thread_id')['role'].value_counts().unstack().fillna(0)
df_threads_roles.to_csv('sna_processed_data-threads-roles.csv', header=True)
df_threads_roles.describe()
print(df_threads_roles.head())
df = pd.read_csv('sna_preprocessed.csv')
df = df[['post/comment text', 'thread_id', 'useraccount_id']]
print(df.head())
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['would', 'uhmm'])
df_threads = df.groupby('thread_id')['useraccount_id'].apply(set).reset_index(name='useraccount_ids')
df.post_or_comment_text = df.post_or_comment_text.str.replace("[^A-Za-z ]", " ")
#df.post_or_comment_text = str(df.post_or_comment_text)
#df.post_or_comment_text = df.post_or_comment_text.apply(cleaning)
# Cleaning Function
def cleaning(string):
    string = string.lower()
    string = string.replace('àö', '')
    string = string.replace('çå', '')
    string = string.replace('ç', '')
    string = string.replace('å', '')
    string = string.replace('äôs', '')
    string = string.replace('çç', '')
    string = string.replace('äôt', '')
    string = string.lower()
   
    from nltk.stem import WordNetLemmatizer
    import nltk
    words = nltk.word_tokenize(string)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(i) for i in words]
    new_words = []
    for i in words:
        if i.isdigit():
            new_words.append('NUMBER')
        else:
            new_words.append(i)
    return ' '.join(new_words)
    df_threads_text = df.groupby('thread_id')['post/comment text'].apply(set).reset_index(name='post/comment texts')
    df_threads_userid = df.groupby('thread_id')['useraccount_id'].apply(set).reset_index(name='useraccount_ids')
    print(df_threads_text.head())
    df_threads_userid.to_csv('sna_processed_data-users-in-thread.csv', header=True)
    print(df_threads_userid.head())
df_threads_text['length'] = df_threads_text['post/comment texts'].apply(len)
df_threads_text = df_threads_text[df_threads_text['length'] > 1]
print(df_threads_text.head())
df_threads_text.to_csv('threads_texts_0917.csv', header=True)