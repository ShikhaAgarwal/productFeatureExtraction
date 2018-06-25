import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import json

#filename = "reviews_Cell_Phones_and_Accessories_5.json"
#with open(filename, 'r') as f:
 #   df = json.load(f)
#print(df['reviewText'])

df = pd.read_json('reviews_Cell_Phones_and_Accessories_5.json', lines=True)
count = 0
#df['reviewText'] = df.reviewText.to_series().astype(str)
reviews = df['reviewText']
print(type(reviews))
output = open("output.txt", "wb")
for col in reviews:
    tokens = nltk.word_tokenize(col)
    tagged = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tagged if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    downcased = [x.lower() for x in nouns]
    joined = " ".join(downcased).encode('utf-8')
    #into_string = str(nouns)
    output.write(joined)
output.close()
