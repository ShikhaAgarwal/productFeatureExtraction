import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice
import numpy as np

#filename = "reviews_Cell_Phones_and_Accessories_5.json"
#with open(filename, 'r') as f:
 #   df = json.load(f)
#print(df['reviewText'])

df = pd.read_json('reviews_Cell_Phones_and_Accessories_5.json', lines=True)
df1 = df.groupby('asin', as_index=False).agg(lambda x: x.tolist())
reviews = df1['reviewText']
asin_list = df1['asin']
NOUNS = ["NN", "NNS"]

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    # topn_ids = np.argsort(row)[::-1][:top_n]
    # top_feats = [(features[i], row[i]) for i in topn_ids]
    # df = pd.DataFrame(top_feats)
    # df.columns = ['feature', 'tfidf']
    # return df

    sorted_ids = np.argsort(row)[::-1]
    tagged = nltk.pos_tag(features)
    count = 0
    top_feats = []
    for i in sorted_ids:
        if tagged[i][1] in NOUNS:
            top_feats.append((features[i], row[i]))
            count += 1
        if count == top_n:
            break
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
	''' Top tfidf features in specific document (matrix row) '''
	row = np.squeeze(Xtr[row_id].toarray())
	return top_tfidf_feats(row, features, top_n)

tvec = TfidfVectorizer(min_df=1, max_df=.5, stop_words='english', ngram_range=(1,2))
tvec_weights = tvec.fit_transform(df['reviewText'])
# vec = tvec.named_steps['vec']
features = tvec.get_feature_names()

output = open("output.csv", "wb")
rows = tvec_weights.shape[0]
for row_id, asin in zip(range(rows), asin_list):
    a = top_feats_in_doc(tvec_weights, features, row_id, top_n=5)
    joined = ", ".join(a.feature).encode('utf-8')
    head = asin + ": "
    output.write(head.encode('utf-8'))
    output.write(joined)
    output.write("\n".encode('utf-8'))
output.close()

#---------Tf-idf----------

# output = open("output.csv", "wb")
# for col, asin in zip(reviews, asin_list):
#     tvec = TfidfVectorizer(min_df=1, max_df=.5, stop_words='english', ngram_range=(1,2))
#     tvec_weights = tvec.fit_transform(col)
#     #print(len(tvec.vocabulary_))
#     weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
#     weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
#     terms = weights_df.sort_values(by='weight', ascending=False).head(5)['term']
#     features = [t.lower() for t in terms]
#     joined = ", ".join(features).encode('utf-8')
#     #print(joined)
#     head = asin + ": "
#     output.write(head.encode('utf-8'))
#     output.write(joined)
#     output.write("\n".encode('utf-8'))
# output.close()

#-------POS----------

# reviews = df['reviewText']
# output = open("output.txt", "wb")
# for col in reviews:
#     tokens = nltk.word_tokenize(col)
#     tagged = nltk.pos_tag(tokens)
#     nouns = [word for word,pos in tagged if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
#     downcased = [x.lower() for x in nouns]
#     joined = " ".join(downcased).encode('utf-8')
#     output.write(joined)
# output.close()
