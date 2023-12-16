import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import text2emotion as te
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# nltk.download('all')


def nltk_text_processing():
    content = ""
    
    with open("./bbc.txt", 'r') as file:
        content = file.read()

    stpWords = stopwords.words('english')
    stpWords.extend(["p", "dc", "mr"])

    tokens = word_tokenize(content.lower())
    print(len(tokens))

    tokens_without_punctuation = [token for token in tokens if token.isalpha()]
    filtered_tokens = [token for token in tokens_without_punctuation if token not in stpWords]
    print(len(filtered_tokens))

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    print(len(lemmatized_tokens))

    words = nltk.Text(lemmatized_tokens)
    fd = words.vocab()
    word = [data[0] for data in fd.most_common()[:10]]
    amount =  [data[1] for data in fd.most_common()[:10]]

    _, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    ax.bar(word, amount, color=colors)
    ax.set_ylabel("Word Frequency")
    ax.set_title("Word amount in text")
    plt.show()

    wordcloud = WordCloud().generate(" ".join(filtered_tokens))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def opinion_assesment():
    goodReview = """I was so happy by you. Thank you so much! 
    The Breakfast was so good. But the Location of the Hotel is a little far away 
    from the Center of the City, this thing made the Visiting of the entire City a little difficult.
    But in general it was nice!"""

    badReview = """I will never stay there again and will not recommend to other people. 
    The overall experince was terrible. I will never recommend it to someone.
    Thin walls, noises from other rooms. One could clearly hear men urinating in the bathroom. 
    Constant noise coming from room upstairs like someone was moving furniture around plus loud parties until 3 am.
    I have requested a quiet room and was given room next to housekeeping store, so every day at 7:30 am housekeepers 
    loudly pulled out their equipment banging it at the walls. So not much for sleeping considering parties going on until 3am.
    Never will stay there again."""


    sid = SentimentIntensityAnalyzer()
    good_rev = sid.polarity_scores(goodReview)
    print(f"Good Review Result: {good_rev}")
    bad_rev = sid.polarity_scores(badReview)
    print(f"Bad Review Result: {bad_rev}\n")

    good_emotion = te.get_emotion(goodReview)
    print(f"Good Emotion: {good_emotion}\n")
    bad_emotion = te.get_emotion(badReview)
    print(f"Bad Emotion: {bad_emotion}")

if __name__ == '__main__':
    nltk_text_processing()    
    opinion_assesment()
    