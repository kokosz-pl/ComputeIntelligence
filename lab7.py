from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('all')



if __name__ == '__main__':
    
    content = ""
    
    with open("./bbc.txt", 'r') as file:
        content = file.read()
        
    tokens = word_tokenize(content.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    print(filtered_tokens)

            