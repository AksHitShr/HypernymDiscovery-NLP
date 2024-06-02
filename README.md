# Introduction to NLP Project

Link for task description: <href> https://competitions.codalab.org/competitions/17119 </href>

> `python3 corpus_preprocess.py [subtask path-corpus dir-datasets path-output]`     
This takes the corpus, the dataset directory and the subtask, it outputs the processed corpus after replacing trigrams and bigrams with underscore separated words etc.
---
> `word2vec/trunk/word2vec -train [path-preprocessed-corpus] -read-vocab [path-preprocessed-corpus].vocab -output [path-output] -cbow 0`        
This trains the embeddings on the processed corpus, and saves the vocabulary associated with this and also the embeddings associated with this.
---
> `python3 prep_data.py [subtask dir-datasets path-embeddings path-output]`     
This takes the corpus, the dataset directory, the subtask and path to the embeddings, outputs the processed data after replacing trigrams and bigrams with underscore separated words etc, generating mappings of embeddings with words and other processing.
---
> `python3 path/to/SemEval-Task9/task9-scorer.py path_to ground_truth path_to_predicted`      
Script provided in the Semeval task itself, in order to evaluate the models/
---
> `hearst.ipynb`        
This takes the tokenized corpus and provides the embeddings based on a rule based approach as mentioned in the report. The final hyponyms are stored in a file called output.txt
---
> `nearestneighbour.ipynb`  
This is an implementation of the nearest neighbours approach as mentioned in the report. It takes the embeddings and the dataset itself, and outputs the hypernyms in a file called 'NN_predicted_processed.txt'
---
> `neuralnetwork.py`
This is an implementation of the neural networks approach. It trains and stores a model for a given number of negative samples 
---
> `neuralnetwork.ipynb`
This trains a neural network for given number of negative samples. Moreover, it also runs inference and saves the output in a file.
---
> `LLM.ipynb`
This is the code which uses Gemini API to predict the hypernyms
---
