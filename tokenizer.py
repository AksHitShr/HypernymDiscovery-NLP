import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')  # Download the tokenizer models if you haven't already
tokenizer = nltk.RegexpTokenizer(r"\w+")

# Function to tokenize text from a file
def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentences = sent_tokenize(text)  # Tokenize text into sentences
        tokenized_sentences = []
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence.lower())  # Tokenize sentence into words
            # Remove punctuation tokens
            tokens = [token for token in tokens if token not in string.punctuation]
            # Join tokens into a single string
            tokenized_sentence = ' '.join(tokens)
            tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences

# Function to process files in a directory
def process_directory(directory):
    tokenized_texts = []
    count = 0 
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Assuming UMBC corpus files have .txt extension
            file_path = os.path.join(directory, filename)
            tokenized_sentences = tokenize_file(file_path)
            tokenized_texts.extend(tokenized_sentences)  # Extend list of tokenized texts
            count += 1
            print(count)
    return tokenized_texts

# Main function
def main():
    # Directory containing UMBC corpus files
    corpus_directory = 'webbase_all'

    # Process files in the directory
    tokenized_texts = process_directory(corpus_directory)

    # Write tokenized texts to a single file
    output_file = '/home/tokenized_umbc_corpus_sentences.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        for tokenized_text in tokenized_texts:
            file.write(tokenized_text + '\n')  # Add two new lines after each sentence

    print(f"Tokenized texts saved to {output_file}")

if __name__ == "__main__":
    main()
