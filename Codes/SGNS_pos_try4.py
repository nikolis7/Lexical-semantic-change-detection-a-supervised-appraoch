import os
import re
from string import punctuation
from collections import OrderedDict
import unidecode
import numpy as np
import logging
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import pandas as pd
from numpy.linalg import svd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity as cos 
from numpy import dot
from numpy.linalg import norm
from gensim.models.word2vec import PathLineSentences
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import punkt 
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import multiprocessing
import gensim.downloader as api
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from scipy.spatial.distance import jensenshannon
from gensim.similarities import WmdSimilarity
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


















#################################   START   #################################


cores = multiprocessing.cpu_count() # Count the number of cores in a computer


punctuation = list(punctuation)
stop_words = stopwords.words('english')


class Data():

    def __getitem__(self, path=None):
        if path != None:
            with open(path, 'r') as f:
                self.doc = f.read()
            return self.doc

    def _preprocess(self, targets, corpus):
        self.index = []
        self.t_index = OrderedDict()
        for target in targets:

            for _, item in enumerate(corpus):
                if target in item:
                    count_target = item.count(target)
                    # Avoiding the sentences with multiple occurrences of the target term for the time being###
                    if count_target == 1:
                        if target not in self.t_index.keys():
                            self.t_index[target] = [_]
                        else:
                             self.t_index[target].append(_)
                        self.index.append(_)
        return self.index, self.t_index


'''
LOAD & EXTRACT DATA
'''
root_dir = os.getcwd()


p1 = os.path.join(root_dir, 'ccoha1.txt')
p2 = os.path.join(root_dir, 'ccoha2.txt')
t = os.path.join(root_dir, 'targets.txt')
r = os.path.join(root_dir, 'targets_results_.txt')


datasets = Data()  # initialization
doc1 = datasets.__getitem__(p1).split('\n')
doc2 = datasets.__getitem__(p2).split('\n')
t1 = datasets.__getitem__(t).split('\n')
results = datasets.__getitem__(r).split('\n')
target_act = [x for x in t1 if len(x) > 1]
t1 = [x.lower() for x in t1 if len(x) > 1]
index1 = datasets._preprocess(t1, doc1)
index2 = datasets._preprocess(t1, doc2)
index_t1 = index1[1]
index_t2 = index2[1]
print('The target words are:', t1)
target_words = t1
targets = target_words


concatenated_list = list(zip(t1, results))

concatenated_list

len(results)

len(targets)


def remove_pos_tags(corpus):
    # Define the suffixes to remove
    suffixes = ('_nn', '_vb')
    
    # Iterate over each sentence in the corpus
    cleaned_corpus = []
    for sentence in corpus:
        # Split the sentence into words and remove suffixes if present
        cleaned_sentence = " ".join(word[:-3] if word.endswith(suffixes) else word for word in sentence.split())
        cleaned_corpus.append(cleaned_sentence)
    
    return cleaned_corpus



doc1_clean = remove_pos_tags(doc1)
doc2_clean = remove_pos_tags(doc2)



# conversions
target_uni = [unidecode.unidecode(m) for m in t1]

t
# Clean corpus 
def listcorpus(doc):
    corpus = []
    for sentence in doc:
        corpus.append([token for token in word_tokenize(sentence)
                          if token not in stop_words and token not in punctuation])
    return corpus


corpus1 = listcorpus(doc1)
corpus2 = listcorpus(doc2)



#############################
#Clean dataset from POS

def remove_pos(corpus): 
    clean_corpus = []
    for sentence in corpus: 
        sent = [] 
        for word in sentence: 
            if word.endswith('_nn') or word.endswith('_vb'):
                sent.append(word.split('_')[0])
            else:
                sent.append(word) 
        clean_corpus.append(sent)
    return clean_corpus
   



corpus1_ = remove_pos(corpus1)
corpus2_ = remove_pos(corpus2)


doc2[25:30]
########################
# Useful function 

# Create a matrix with the word embeddings
def matrix(words, model):
    x = {}
    for word in words:
        print(word)
        x[word] = model.wv[word]

    p = pd.DataFrame(x)
    return p


# Orthogonal Procrustes 
def OP(embeddings_corpus_1, embeddings_corpus_2):

    # Orthogonal Procrustes
    A = embeddings_corpus_1.to_numpy()
    B = embeddings_corpus_2.to_numpy()

    # Orthogonal Procrustes is used to find a matrix R which will allow for a mapping from A to B
    M = np.dot(B, A.transpose())
    u, s, vh = svd(M)
    R = np.dot(u, vh)

    # Transform A using R to map it to B. The transformed matrix A_new = RA
    # Mapped matrix
    new_A = np.dot(R, A)

    return new_A, B


# Orthogonal Procrustes scipy 
def OP_2(embeddings_corpus_1, embeddings_corpus_2):

    # Orthogonal Procrustes
    A = embeddings_corpus_1.to_numpy()
    B = embeddings_corpus_2.to_numpy()

    R, sca = orthogonal_procrustes(A, B)
    
    new_A = np.dot(A, R.T)
    
    return new_A, B



# Cosine Similarity 
def cosine_similarity(target_words, new_A, B):

    output = {}

    for i, word in enumerate(target_words):
        output[word] = dot(new_A[:, i].transpose(), B[:, i].transpose(
        ))/(norm(new_A[:, i].transpose())*norm(B[:, i].transpose()))

    return output



def classify(output):
    s = []
    for i, j in output.items():
        if j > 0.92:
            s.append(0)
        else:
            s.append(1)
    return s



def classify(output, threshold):
    s = []
    for word, similarity in output.items():
        if similarity > threshold:
            s.append(0)  # No change
        else:
            s.append(1)  # Change
    return s





import numpy as np
import scipy.stats as stats

def classify(output, confidence=0.95):
    # Convert output values to a numpy array for calculations
    similarities = np.array(list(output.values()))
    
    # Calculate the mean of cosine similarities
    mean_similarity = np.mean(similarities)
    
    # Calculate the confidence interval
    confidence_interval = stats.sem(similarities) * stats.t.ppf((1 + confidence) / 2., len(similarities) - 1)
    
    # Set the threshold as mean similarity minus the confidence interval
    threshold = mean_similarity - confidence_interval
    print(f"Threshold for classification (mean - CI): {threshold}")
    
    # Classify based on the new threshold
    s = [0 if similarity > threshold else 1 for similarity in similarities]
    
    return s





from scipy import stats
import numpy as np

def classify_m(output, threshold_type="mean_ci", confidence=0.9, verbose=False):
    """
    Classify words based on cosine similarity and the specified threshold type.
    
    :param output: Dictionary of words and their cosine similarities.
    :param threshold_type: Type of threshold to use. Options: "mean", "mean_ci", "median", "5th_percentile".
    :param confidence: Confidence level for "mean_ci" (default is 0.9).
    :param verbose: If True, prints the threshold used for classification.
    :return: List of classifications (0 or 1) for each word.
    """
    # Convert output values to a numpy array for calculations
    similarities = np.array(list(output.values()))

    # Check for empty input
    if len(similarities) == 0:
        raise ValueError("The `output` dictionary is empty, cannot classify.")
    
    # Determine the threshold based on the selected type
    if threshold_type == "mean":
        threshold = np.mean(similarities)
    elif threshold_type == "mean_ci":
        confidence_interval = stats.sem(similarities) * stats.t.ppf((1 + confidence) / 2., len(similarities) - 1)
        threshold = np.mean(similarities) - confidence_interval
    elif threshold_type == "median":
        threshold = np.median(similarities)
    elif threshold_type == "5th_percentile":
        threshold = np.percentile(similarities, 5)
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    # Debugging: Print threshold if verbose
    if verbose:
        print(f"Threshold for classification ({threshold_type}): {threshold}")
    
    # Classify based on the threshold
    classifications = [0 if similarity > threshold else 1 for similarity in similarities]
    
    return classifications





def accuracy(s, results, output):
    count = 0
    for i, word in enumerate(output):
        if s[i] == int(results[i]):
            count += 1
    acc = count / len(results)
    return acc




##########################################Import dataset with pos########################################################




# function to load the text corpus
def load_corpus_from_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip().split() for line in file]
    

# Load the corpus
corpus_1_pos = load_corpus_from_text("processed_corpus_1.txt")
corpus_2_pos = load_corpus_from_text("processed_corpus_2.txt")





             
# Crreate a function for finding the all common words

def extract_words(corpus):
    """Helper function to extract words from a corpus."""
    words = set()
    for sentence in corpus:
        for word in sentence:
            # Here you can add a condition to filter nouns and verbs, if necessary
            words.add(word)
    return words

# Extract nouns and verbs from each corpus
words_corpus1 = extract_words(corpus1)
words_corpus2 = extract_words(corpus2)

# Find the intersection of nouns and verbs between both corpora
common_words = words_corpus1.intersection(words_corpus2)


len(common_words)






def extract_nn_and_vb_words(corpus):
    """
    Extracts words ending with _nn and _vb from a list of POS-tagged sentences.
    
    Parameters:
    corpus (list of lists): POS-tagged tokenized sentences.
    
    Returns:
    list: A combined list of words ending with _nn or _vb.
    """
    words_with_suffix = []
    
    for sentence in corpus:
        for word in sentence:
            if word.endswith('_nn') or word.endswith('_vb'):
                words_with_suffix.append(word)
    
    return words_with_suffix


# Extract words with _nn and _vb suffixes into one list
combined_nn_vb_words_1 = extract_nn_and_vb_words(corpus_1_pos)

combined_nn_vb_words_2 = extract_nn_and_vb_words(corpus_2_pos)



def replace_with_pos_tags_from_list(doc, pos_words_list):
    """
    Replace words in `doc` with their corresponding POS-tagged versions from `pos_words_list`.
    
    Parameters:
    doc (list of str): Original sentences as strings.
    pos_words_list (list of str): List of POS-tagged words (e.g., words ending with _nn or _vb).
    
    Returns:
    list of str: Modified sentences with words replaced by their POS-tagged versions.
    """
    modified_sentences = []
    
    # Create a mapping of base words to their POS-tagged versions from the list
    pos_mapping = {word.split('_')[0]: word for word in pos_words_list}
    
    # Iterate over each sentence in the document
    for sentence in doc:
        # Split the original sentence into words
        sentence_words = sentence.split()
        
        # Replace words in the sentence with POS-tagged versions if available
        modified_sentence = " ".join(pos_mapping.get(word, word) for word in sentence_words)
        modified_sentences.append(modified_sentence)
    
    return modified_sentences



# Replace words in doc1 with POS-tagged versions from combined_nn_vb_words_1
modified_doc1 = replace_with_pos_tags_from_list(doc1, combined_nn_vb_words_1)


modified_doc2 = replace_with_pos_tags_from_list(doc2, combined_nn_vb_words_2)

       
             
             
# Create a function for finding the common words either nn or vb

def get_common_nouns_verbs(corpus1, corpus2):
    """
    Finds common nouns and verbs between two corpora.
    
    Parameters:
    - corpus1: First tokenized corpus with POS-tagged words.
    - corpus2: Second tokenized corpus with POS-tagged words.
    
    Returns:
    - A set of common nouns and verbs (words ending with _nn or _vb) found in both corpora.
    """
    def extract_nouns_verbs(corpus):
        """Helper function to extract nouns and verbs from a corpus."""
        words = set()
        for sentence in corpus:
            for word in sentence:
                # Check if the word is a noun or verb based on suffix
                if word.endswith('_nn') or word.endswith('_vb'):
                    words.add(word)
        return words

    # Extract nouns and verbs from each corpus
    nouns_verbs_corpus1 = extract_nouns_verbs(corpus1)
    nouns_verbs_corpus2 = extract_nouns_verbs(corpus2)
    
    # Find the intersection of nouns and verbs between both corpora
    common_words = nouns_verbs_corpus1.intersection(nouns_verbs_corpus2)
    return common_words



common_nouns_verbs = get_common_nouns_verbs(corpus_1_pos, corpus_2_pos)




# Crreate a function for finding the common words nn 

def get_common_nouns(corpus1, corpus2):
    """
    Finds common nouns and verbs between two corpora.
    
    Parameters:
    - corpus1: First tokenized corpus with POS-tagged words.
    - corpus2: Second tokenized corpus with POS-tagged words.
    
    Returns:
    - A set of common nouns and verbs (words ending with _nn or _vb) found in both corpora.
    """
    def extract_nouns(corpus):
        """Helper function to extract nouns and verbs from a corpus."""
        words = set()
        for sentence in corpus:
            for word in sentence:
                # Check if the word is a noun or verb based on suffix
                if word.endswith('_nn'):
                    words.add(word)
        return words

    # Extract nouns and verbs from each corpus
    nouns_corpus1 = extract_nouns(corpus1)
    nouns_corpus2 = extract_nouns(corpus2)
    
    # Find the intersection of nouns and verbs between both corpora
    common_words = nouns_corpus1.intersection(nouns_corpus2)
    return common_words



common_nouns = get_common_nouns(corpus_1_pos, corpus_2_pos)





# Create a function for finding the common words  vb

def get_common_verbs(corpus1, corpus2):
    """
    Finds common nouns and verbs between two corpora.
    
    Parameters:
    - corpus1: First tokenized corpus with POS-tagged words.
    - corpus2: Second tokenized corpus with POS-tagged words.
    
    Returns:
    - A set of common nouns and verbs (words ending with _nn or _vb) found in both corpora.
    """
    def extract_verbs(corpus):
        """Helper function to extract nouns and verbs from a corpus."""
        words = set()
        for sentence in corpus:
            for word in sentence:
                # Check if the word is a noun or verb based on suffix
                if  word.endswith('_vb'):
                    words.add(word)
        return words

    # Extract nouns and verbs from each corpus
    verbs_corpus1 = extract_verbs(corpus1)
    verbs_corpus2 = extract_verbs(corpus2)
    
    # Find the intersection of nouns and verbs between both corpora
    common_words = verbs_corpus1.intersection(verbs_corpus2)
    return common_words



common_verbs = get_common_verbs(corpus_1_pos, corpus_2_pos)




len(common_nouns)
len(common_verbs)
len(common_nouns_verbs)

len(common_nouns)+len(common_verbs)




##############################################################################################################################################################################
##############################################################################################################################################################################


############################ Found common noouns in the two dataset with more than 3 appereances in both datasets ############################




import pandas as pd
from collections import Counter

def find_common_nouns(corpus_1, corpus_2, min_occurrences=3, output_file="common_nouns.csv"):
    # Helper function to extract nouns from a corpus
    def extract_nouns(corpus):
        nouns = []
        for sentence in corpus:
            nouns.extend([word for word in sentence if word.endswith('_nn')])
        return nouns
    
    # Count nouns in each corpus
    nouns_corpus_1 = Counter(extract_nouns(corpus_1))
    nouns_corpus_2 = Counter(extract_nouns(corpus_2))
    
    # Find common nouns that appear in at least `min_occurrences` sentences in each corpus
    common_nouns = {
        noun: (nouns_corpus_1[noun], nouns_corpus_2[noun])
        for noun in nouns_corpus_1
        if noun in nouns_corpus_2 and nouns_corpus_1[noun] >= min_occurrences and nouns_corpus_2[noun] >= min_occurrences
    }
    
    # Convert the results to a DataFrame for easier CSV export
    results_df = pd.DataFrame.from_dict(common_nouns, orient='index', columns=['Corpus 1 Count', 'Corpus 2 Count']).reset_index()
    results_df = results_df.rename(columns={'index': 'Noun'})
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)
    
    print(f"Common nouns saved to {output_file}")
    return results_df

# Example usage
common_nouns_df = find_common_nouns(corpus_1_pos, corpus_2_pos)


common_nouns_df.describe


common_nouns_df.shape





##############################################################################################################################################################################
##############################################################################################################################################################################

############################################################ SGNS PIPELINE ###################################################################################################

def sgns_pipeline(corpus1, corpus2, targets, results, pretrained=False, alignment="OP", epochs=10, plot=True):
    
    # Initialize output to avoid UnboundLocalError
    output = None
    results_df = None  # Initialize a DataFrame to store the results

    # Define number of CPU cores available
    cores = multiprocessing.cpu_count()

    if not pretrained:
        # Initialize SGNS model for corpus1
        model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                           sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
        # Build vocabulary and train the model
        model_1.build_vocab(corpus1, progress_per=10000)
        model_1.train(corpus1, total_examples=model_1.corpus_count, epochs=epochs)
        model_1.save(f"model_1_no_pretrained_epochs_{epochs}.model")
        
        # Get embeddings from corpus1
        embeddings_corpus_1 = matrix(targets, model_1)
        
        if alignment == "OP":
            # Initialize and train the second SGNS model for corpus2
            model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                               sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
            model_2.build_vocab(corpus2, progress_per=10000)
            model_2.train(corpus2, total_examples=model_2.corpus_count, epochs=epochs)
            model_2.save(f"model_2_no_pretrained_epochs_{epochs}.model")
            
            # Get embeddings from corpus2
            embeddings_corpus_2 = matrix(targets, model_2)
            
            # Alignment with Orthogonal Procrustes
            new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
            output = cosine_similarity(targets, new_A, B)
            
        else:
            # Incremental learning on corpus2
            model_1.train(corpus2, total_examples=model_1.corpus_count, epochs=epochs)
            model_1.save(f"model_1_incremental_epochs_{epochs}.model")
            
            # Get embeddings from corpus2
            embeddings_corpus_2 = matrix(targets, model_1)
            output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())

    else:
        # Load pre-trained model
        pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
        # Initialize and fine-tune model on corpus1
        model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                         sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
        model.build_vocab(corpus1, progress_per=10000)
        
        # Copy pre-trained vectors
        for word in pretrained_model.index_to_key:
            if word in model.wv:
                model.wv[word] = pretrained_model[word]
                
        model.train(corpus1, total_examples=model.corpus_count, epochs=epochs, start_alpha=0.0001, end_alpha=0.0001)
        model.save(f"fine_tuned_model_1_epochs_{epochs}.model")
        
        embeddings_corpus_1 = matrix(targets, model)
        
        if alignment == "OP":
            # Initialize model_2 and fine-tune for corpus2
            model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                               sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
            model_2.build_vocab(corpus2, progress_per=10000)
            
            for word in pretrained_model.index_to_key:
                if word in model_2.wv:
                    model_2.wv[word] = pretrained_model[word]
                    
            model_2.train(corpus2, total_examples=model_2.corpus_count, epochs=epochs, start_alpha=0.0001, end_alpha=0.0001)
            model_2.save(f"fine_tuned_model_2_epochs_{epochs}.model")
            
            embeddings_corpus_2 = matrix(targets, model_2)
            new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
            output = cosine_similarity(targets, new_A, B)
            
        else:
            # Incremental learning for pre-trained model on corpus2
            model.build_vocab(corpus2, update=True)
            model.train(corpus2, total_examples=model.corpus_count, epochs=epochs, start_alpha=0.0001, end_alpha=0.0001)
            model.save(f"fine_tuned_incremental_model_1_epochs_{epochs}.model")
            
            embeddings_corpus_2 = matrix(targets, model)
            output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())

    if output:
        # Classify and calculate accuracy
        s = classify(output)
        acc = accuracy(s, results, output)
        print(f"Accuracy: {acc}")
        
        # Save results to DataFrame
        results_df = pd.DataFrame({
            'Word': list(output.keys()),
            'Cosine Similarity': list(output.values()),
            'Classification': s
        })
        results_df.to_csv("sgns_pipeline_results.csv", index=False)
        
        # Plot if requested
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(output.values(), output.values(), label="Cosine Similarity")
            for (x, y) in output.items():
                ax.text(y, y, x, va='bottom', ha='center')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Words')
            ax.get_yaxis().set_visible(False)
            ax.legend()
            plt.show()
        
        # Calculate and print average similarity threshold
        avg_similarity_threshold = threshold(output, results)
        print(f"Average Cosine Similarity Threshold for Semantic Change: {avg_similarity_threshold}")
        return avg_similarity_threshold, results_df, 'Run completed!'

    print("Output is None, threshold calculation skipped.")
    return None, results_df, 'Run completed!'



##########################################################################################################################
# Try pipeline 
sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=False, alignment="OP", epochs=100, plot=True)

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=False, alignment="OP", epochs=200, plot=True)

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=False, alignment="OP", epochs=300, plot=True)


#########################################################################################################################


########################################################################################################################
# Try pipeline 
sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=False, alignment="incremental", epochs=100, plot=True)

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=False, alignment="incremental", epochs=200, plot=True)

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=False, alignment="incremental", epochs=300, plot=True)

########################################################################################################################


#######################################################################################################################
# Try pipeline 
sgns_pipeline(targets=targets, pretrained=True, alignment="OP", epochs=100, plot=True)

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=True, alignment="OP", epochs=300, plot=True)


#######################################################################################################################


######################################################################################################################
# Try pipeline 
sgns_pipeline(targets=targets, pretrained=True, alignment="incremental", epochs=10, plot=True)
######################################################################################################################

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=True, alignment="incremental", epochs=100, plot=True)

sgns_pipeline(corpus_1_pos, corpus_2_pos, targets, results, pretrained=True, alignment="incremental", epochs=300, plot=True)


############################################################################################################################################################################################
############################################################################################################################################################################################




########################################################## Evaluate Results ##################################################################################################################################

'''
def evaluate_saved_models(model_path1, model_path2, targets, results, concatenated_list, alignment="OP", plot=True, confidence=0.95):
    # Load the saved models
    model_1 = Word2Vec.load(model_path1)
    model_2 = Word2Vec.load(model_path2)
    
    # Filter target words to ensure they exist in both models
    targets_filtered = [word for word in targets if word in model_1.wv and word in model_2.wv]
    missing_words = set(targets) - set(targets_filtered)
    
    if missing_words:
        print(f"Warning: The following target words are missing in the model vocabulary and will be ignored: {missing_words}")
    
    # Get embeddings for the filtered target words
    embeddings_corpus_1 = matrix(targets_filtered, model_1)
    embeddings_corpus_2 = matrix(targets_filtered, model_2)
    
    # Perform alignment if specified
    if alignment == "OP":
        new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
        output = cosine_similarity(targets_filtered, new_A, B)
    else:
        # If alignment is not Orthogonal Procrustes, use embeddings directly
        output = cosine_similarity(targets_filtered, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
    
    # Classify words based on the dynamic threshold (mean - confidence interval)
    classification = classify(output, confidence=confidence)
    
    # Calculate accuracy by comparing to provided `results`
    acc = accuracy(classification, results, output)
    print(f"Accuracy: {acc}")
    
    # Create a dictionary for ground truth lookup from concatenated_list
    ground_truth_dict = dict(concatenated_list)
    
    # Save results to DataFrame
    results_df = pd.DataFrame({
        'Word': list(output.keys()),
        'Cosine Similarity': list(output.values()),
        'Classification': classification,
        'Ground Truth': [ground_truth_dict.get(word, 'N/A') for word in output.keys()]  # Retrieve ground truth from the dictionary
    })
    
    # Plot if requested
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(output.values(), output.values(), label="Cosine Similarity")
        for word, similarity in output.items():
            ax.text(similarity, similarity, word, va='bottom', ha='center')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Words')
        ax.get_yaxis().set_visible(False)
        ax.legend()
        plt.show()
    
    # Return the results DataFrame and accuracy
    return results_df, acc
'''






'''

def matrix(words, model):
    return np.array([model.wv[word] for word in words])

def OP(embeddings_1, embeddings_2):
    u, _, vt = np.linalg.svd(embeddings_2.T @ embeddings_1)
    rotation = u @ vt
    new_embeddings_1 = embeddings_1 @ rotation
    return new_embeddings_1, embeddings_2

def cosine_similarity(words, emb1, emb2):
    return {word: np.dot(emb1[i], emb2[i]) / (np.linalg.norm(emb1[i]) * np.linalg.norm(emb2[i]))
            for i, word in enumerate(words)}

def accuracy(predictions, ground_truth, output):
    return np.mean([int(pred == int(gt)) for pred, gt in zip(predictions, ground_truth)])


'''






import numpy as np
from scipy.linalg import svd

def OP(embeddings_corpus_1, embeddings_corpus_2):
    """
    Perform Orthogonal Procrustes alignment on two embedding matrices.
    
    :param embeddings_corpus_1: Embedding matrix for corpus 1 (Pandas DataFrame or numpy array).
    :param embeddings_corpus_2: Embedding matrix for corpus 2 (Pandas DataFrame or numpy array).
    :return: Tuple (new_A, B) where new_A is the transformed matrix for corpus 1 and B is the matrix for corpus 2.
    """
    # Convert to numpy arrays if inputs are DataFrames
    if not isinstance(embeddings_corpus_1, np.ndarray):
        A = embeddings_corpus_1.to_numpy()
    else:
        A = embeddings_corpus_1

    if not isinstance(embeddings_corpus_2, np.ndarray):
        B = embeddings_corpus_2.to_numpy()
    else:
        B = embeddings_corpus_2

    # Check for dimensionality mismatch
    if A.shape[1] != B.shape[1]:
        raise ValueError("Embedding matrices A and B must have the same number of dimensions.")

    # Orthogonal Procrustes
    M = np.dot(B.T, A)  # Compute dot product B * A^T
    u, _, vh = svd(M)
    R = np.dot(u, vh)  # Compute rotation matrix

    # Transform A using R
    new_A = np.dot(A, R.T)  # Transpose R for correct alignment

    return new_A, B

















from scipy import stats
import numpy as np

def classify_m(output, threshold_type="mean_ci", confidence=0.9, verbose=False):
    """
    Classify words based on cosine similarity and the specified threshold type.
    
    :param output: Dictionary of words and their cosine similarities.
    :param threshold_type: Type of threshold to use. Options: "mean", "mean_ci", "median", "5th_percentile".
    :param confidence: Confidence level for "mean_ci" (default is 0.9).
    :param verbose: If True, prints the threshold used for classification.
    :return: List of classifications (0 or 1) for each word.
    """
    # Convert output values to a numpy array for calculations
    similarities = np.array(list(output.values()))

    # Check for empty input
    if len(similarities) == 0:
        raise ValueError("The `output` dictionary is empty, cannot classify.")
    
    # Determine the threshold based on the selected type
    if threshold_type == "mean":
        threshold = np.mean(similarities)
    elif threshold_type == "mean_ci":
        confidence_interval = stats.sem(similarities) * stats.t.ppf((1 + confidence) / 2., len(similarities) - 1)
        threshold = np.mean(similarities) - confidence_interval
    elif threshold_type == "median":
        threshold = np.median(similarities)
    elif threshold_type == "5th_percentile":
        threshold = np.percentile(similarities, 5)
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    # Debugging: Print threshold if verbose
    if verbose:
        print(f"Threshold for classification ({threshold_type}): {threshold}")
    
    # Classify based on the threshold
    classifications = [0 if similarity > threshold else 1 for similarity in similarities]
    
    return classifications




def evaluate_saved_models(model_path1, model_path2, targets, results, concatenated_list, alignment="OP", threshold_type="mean_ci", plot=True, confidence=0.95):
    """
    Evaluate saved models and classify words based on semantic change.
    """
    # Load the saved models
    model_1 = Word2Vec.load(model_path1)
    model_2 = Word2Vec.load(model_path2)
    
    # Filter target words to ensure they exist in both models
    targets_filtered = [word for word in targets if word in model_1.wv and word in model_2.wv]
    missing_words = set(targets) - set(targets_filtered)
    
    if missing_words:
        print(f"Warning: The following target words are missing in the model vocabulary and will be ignored: {missing_words}")
    
    # Get embeddings for the filtered target words
    embeddings_corpus_1 = matrix(targets_filtered, model_1)
    embeddings_corpus_2 = matrix(targets_filtered, model_2)
    
    # Ensure embeddings are NumPy arrays
    if not isinstance(embeddings_corpus_1, np.ndarray):
        embeddings_corpus_1 = embeddings_corpus_1.to_numpy()
    if not isinstance(embeddings_corpus_2, np.ndarray):
        embeddings_corpus_2 = embeddings_corpus_2.to_numpy()
    
    # Perform alignment if specified
    if alignment == "OP":
        new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
        output = cosine_similarity(targets_filtered, new_A, B)
    else:
        # Directly use embeddings if no alignment is needed
        output = cosine_similarity(targets_filtered, embeddings_corpus_1, embeddings_corpus_2)
    
    # Classify words based on the specified threshold type
    classification = classify_m(output, threshold_type=threshold_type, confidence=confidence)
    
    # Calculate accuracy by comparing to provided `results`
    acc = accuracy(classification, results, output)
    print(f"Accuracy: {acc}")
    
    # Create a dictionary for ground truth lookup from concatenated_list
    ground_truth_dict = dict(concatenated_list)
    
    # Save results to DataFrame
    results_df = pd.DataFrame({
        'Word': list(output.keys()),
        'Cosine Similarity': list(output.values()),
        'Classification': classification,
        'Ground Truth': [ground_truth_dict.get(word, 'N/A') for word in output.keys()]
    })
    
    # Plot if requested
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(output.values(), output.values(), label="Cosine Similarity")
        for word, similarity in output.items():
            ax.text(similarity, similarity, word, va='bottom', ha='center')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Words')
        ax.get_yaxis().set_visible(False)
        ax.legend()
        plt.show()
    
    # Return the results DataFrame and accuracy
    return results_df, acc






# Check results

model_path1 = "fine_tuned_model_1_epochs_300.model"
model_path2 = "fine_tuned_incremental_model_1_epochs_300.model"

results_df, acc = evaluate_saved_models(
    model_path1=model_path1,
    model_path2=model_path2,
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,
    alignment="OP",
    threshold_type="median",
    plot=True,
    confidence=0.8
)


results_df.to_csv("test.csv", index=False)



targets
results
concatenated_list








############################################################# Print words with threshold ###########################################################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_saved_models(model_path1, model_path2, targets, results, concatenated_list, alignment="OP", threshold_type="mean_ci", plot=True, confidence=0.95):
    """
    Evaluate saved models and classify words based on semantic change.
    """
    # Load the saved models
    model_1 = Word2Vec.load(model_path1)
    model_2 = Word2Vec.load(model_path2)
    
    # Filter target words to ensure they exist in both models
    targets_filtered = [word for word in targets if word in model_1.wv and word in model_2.wv]
    missing_words = set(targets) - set(targets_filtered)
    
    if missing_words:
        print(f"Warning: The following target words are missing in the model vocabulary and will be ignored: {missing_words}")
    
    # Get embeddings for the filtered target words
    embeddings_corpus_1 = matrix(targets_filtered, model_1)
    embeddings_corpus_2 = matrix(targets_filtered, model_2)
    
    # Ensure embeddings are NumPy arrays
    if not isinstance(embeddings_corpus_1, np.ndarray):
        embeddings_corpus_1 = embeddings_corpus_1.to_numpy()
    if not isinstance(embeddings_corpus_2, np.ndarray):
        embeddings_corpus_2 = embeddings_corpus_2.to_numpy()
    
    # Perform alignment if specified
    if alignment == "OP":
        new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
        output = cosine_similarity(targets_filtered, new_A, B)
    else:
        # Directly use embeddings if no alignment is needed
        output = cosine_similarity(targets_filtered, embeddings_corpus_1, embeddings_corpus_2)
    
    # Classify words based on the specified threshold type
    classification = classify_m(output, threshold_type=threshold_type, confidence=confidence)
    
    # Calculate accuracy by comparing to provided `results`
    acc = accuracy(classification, results, output)
    print(f"Accuracy: {acc}")
    
    # Create a dictionary for ground truth lookup from concatenated_list
    ground_truth_dict = dict(concatenated_list)
    
    # Save results to DataFrame
    results_df = pd.DataFrame({
        'Word': list(output.keys()),
        'Cosine Similarity': list(output.values()),
        'Classification': classification,
        'Ground Truth': [ground_truth_dict.get(word, 'N/A') for word in output.keys()]
    })
    
    # Plot if requested
    if plot:
        # Calculate threshold based on the type
        similarities = np.array(list(output.values()))
        if threshold_type == "mean_ci":
            confidence_interval = stats.sem(similarities) * stats.t.ppf((1 + confidence) / 2., len(similarities) - 1)
            threshold = np.mean(similarities) - confidence_interval
        elif threshold_type == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.percentile(similarities, 5)  # Default to 5th percentile for simplicity
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(similarities, np.zeros_like(similarities), color='blue', alpha=0.7, label="Cosine Similarity", s=50)
        
        # Add word labels
        for word, similarity in output.items():
            ax.text(similarity, 0.01, word, fontsize=9, ha='center', va='bottom', rotation=90)
        
        # Plot the threshold line
        ax.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold ({threshold_type})")
        
        # Improve plot aesthetics
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_title('Word Similarities with Threshold Line', fontsize=14)
        ax.legend()
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    
    # Return the results DataFrame and accuracy
    return results_df, acc









model_path1 = "fine_tuned_model_1_epochs_300.model"
model_path2 = "fine_tuned_incremental_model_1_epochs_300.model"

results_df, acc = evaluate_saved_models(
    model_path1=model_path1,
    model_path2=model_path2,
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,
    alignment="Incr",
    threshold_type="median",
    plot=True,
    confidence=0.8
)















############################################################# check results ###########################################################################################


# Example usage:
results_df, accuracy = evaluate_saved_models(
    model_path1="fine_tuned_model_1_epochs_200.model",
    model_path2="fine_tuned_incremental_model_1_epochs_200.model",
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,  # Include concatenated list here
    alignment="incre",
    plot=True,
    confidence=0.85  # Specify the confidence level
)



results_df.to_csv("results_pre_trained_incre_200.csv", index=False)




# 2 Example usage:
results_df, accuracy = evaluate_saved_models(
    model_path1="fine_tuned_model_1_epochs_200.model",
    model_path2="fine_tuned_model_2_epochs_200.model",
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,  # Include concatenated list here
    alignment="OP",
    plot=True,
    confidence=0.85  # Specify the confidence level
)

results_df

results_df.to_csv("results_pre_trained_200.csv", index=False)




# 3 Example usage:
results_df, accuracy = evaluate_saved_models(
    model_path1="model_1_no_pretrained_epochs_200.model",
    model_path2="model_2_no_pretrained_epochs_200.model",
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,  # Include concatenated list here
    alignment="OP",
    plot=True,
    confidence=0.85  # Specify the confidence level
)



results_df.to_csv("results_pre_trained_200.csv", index=False)





# 4 Example usage:
results_df, accuracy = evaluate_saved_models(
    model_path1="model_1_no_pretrained_epochs_100.model",
    model_path2="model_2_no_pretrained_epochs_100.model",
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,  # Include concatenated list here
    alignment="OP",
    plot=True,
    confidence=0.85  # Specify the confidence level
)



results_df.to_csv("results_pre_trained_100.csv", index=False)







# 5 Example usage:
results_df, accuracy = evaluate_saved_models(
    model_path1="fine_tuned_model_1_epochs_300.model",
    model_path2="fine_tuned_incremental_model_1_epochs_300.model",
    targets=targets,
    results=results,
    concatenated_list=concatenated_list,  # Include concatenated list here
    alignment="OP",
    threshold_type="mean_ci",
    plot=True,
    confidence=0.8  # Specify the confidence level
)



results_df.to_csv("results_pre_trained_100.csv", index=False)








############################################################################################################################################################################################
############################################################################################################################################################################################

############################################################################ Check results with all combinations ############################################################################



from scipy.stats import sem, t
import numpy as np




def classify(output, threshold):
    s = []
    for word, similarity in output.items():
        if similarity > threshold:
            s.append(0)  # No change
        else:
            s.append(1)  # Change
    return s












def confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset.
    :param data: List or array of numerical values
    :param confidence: Confidence level (default is 0.95 for 95% CI)
    :return: Confidence interval value
    """
    data = np.array(data)  # Convert to numpy array if not already
    n = len(data)
    if n < 2:
        raise ValueError("At least two data points are required to calculate a confidence interval.")
    mean = np.mean(data)
    std_err = sem(data)  # Standard error of the mean
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)  # t-distribution critical value
    return h



targets



def evaluate_models_with_thresholds_and_alignments(
    model_path1,
    model_path2,
    targets,
    results,
    concatenated_list,
    thresholds=["mean", "mean_ci", "median", "5th_percentile"],
    alignments=["OP", "incremental"],
    confidence=0.95
):
    from collections import defaultdict
    import numpy as np

    # Load the saved models
    model_1 = Word2Vec.load(model_path1)
    model_2 = Word2Vec.load(model_path2)

    # Filter target words to ensure they exist in both models
    targets_filtered = [word for word in targets if word in model_1.wv and word in model_2.wv]
    missing_words = set(targets) - set(targets_filtered)

    if missing_words:
        print(f"Warning: The following target words are missing in the model vocabulary and will be ignored: {missing_words}")

    # Get embeddings for the filtered target words
    embeddings_corpus_1 = matrix(targets_filtered, model_1)
    embeddings_corpus_2 = matrix(targets_filtered, model_2)

    # Initialize results dictionary
    accuracy_results = defaultdict(dict)

    # Loop over alignments and thresholds
    for alignment in alignments:
        # Perform alignment if specified
        if alignment == "OP":
            new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
        elif alignment == "incremental":
            new_A, B = embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy()
        else:
            raise ValueError(f"Unknown alignment method: {alignment}")

        # Calculate cosine similarity
        output = cosine_similarity(targets_filtered, new_A, B)

        # Loop over threshold approaches
        for threshold_type in thresholds:
            # Apply threshold-based classification
            if threshold_type == "mean":
                threshold = np.mean(list(output.values()))
            elif threshold_type == "mean_ci":
                threshold = np.mean(list(output.values())) - confidence_interval(list(output.values()), confidence=confidence)
            elif threshold_type == "median":
                threshold = np.median(list(output.values()))
            elif threshold_type == "5th_percentile":
                threshold = np.percentile(list(output.values()), 5)
            else:
                raise ValueError(f"Unknown threshold type: {threshold_type}")

            # Classify words based on the threshold
            classification = classify(output, threshold=threshold)

            # Calculate accuracy by comparing to provided `results`
            acc = accuracy(classification, results, output)
            accuracy_results[alignment][threshold_type] = acc

            print(f"Alignment: {alignment}, Threshold: {threshold_type}, Accuracy: {acc}")

    return accuracy_results








model_path1 = "model_1_no_pretrained_epochs_300.model"
model_path2 = "model_2_no_pretrained_epochs_300.model"
targets = ["plane_nn", "job_nn", "league_nn", "lot_nn", "car_nn"]
results = {"plane_nn": 1, "job_nn": 0, "league_nn": 1, "lot_nn": 0, "car_nn": 1}  # Example ground truth
concatenated_list = [("plane_nn", 1), ("job_nn", 0), ("league_nn", 1), ("lot_nn", 0), ("car_nn", 1)]

# Call the function
accuracy_results = evaluate_models_with_thresholds_and_alignments(
    model_path1, model_path2, targets, results, concatenated_list,
    thresholds=["mean", "mean_ci", "median", "5th_percentile"],
    alignments=["OP", "incremental"],
    confidence=0.8
)

# Print the results
for alignment, threshold_data in accuracy_results.items():
    for threshold_type, acc in threshold_data.items():
        print(f"Alignment: {alignment}, Threshold: {threshold_type}, Accuracy: {acc}")




















############################################################################################################################################################################################
############################################################################################################################################################################################




############################################################################################################################################################################################
############################################################################################################################################################################################
    
    
    
    
########################################## SELECT THE THRESHOLD FOR COMMON WORDS BASED ON GROUND TRUTH  ##########################################################

#### The methodology is to take the average cosine similarity of the proposed semantic change #######

#Take the best model and find the best threshold 
df = pd.read_csv('results_pre_trained_incre_OP_300.csv')
df1 = pd.read_csv('results_pre_trained_incre_300.csv')



def find_optimal_threshold(df, threshold_range=(0.9, 1.0), step=0.001):
    best_threshold = None
    best_accuracy = 0.0
    
    # Iterate over a range of threshold values
    for threshold in np.arange(*threshold_range, step):
        # Classify as 1 if similarity is below threshold (indicating semantic change)
        df['Predicted Change'] = (df['Cosine Similarity'] < threshold).astype(int)
        
        # Calculate accuracy by comparing with ground truth
        correct_predictions = (df['Predicted Change'] == df['Ground Truth']).sum()
        accuracy = correct_predictions / len(df)
        
        # Update best threshold if current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Print and return results
    print(f"Optimal Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")
    return best_threshold, best_accuracy


# Function application
optimal_threshold, best_accuracy = find_optimal_threshold(df)


##################################### The methodology is to take the average cosine similarity of the proposed semantic change #########################################



# Load data
df = pd.read_csv('results_pre_trained_incre_OP_300.csv')
df1 = pd.read_csv('results_pre_trained_incre_300.csv')

def calculate_average_threshold(df):
    """
    Calculate the average of the 'Cosine Similarity' column as the threshold.
    """
    # Calculate the average cosine similarity
    average_threshold = df['Cosine Similarity'].mean()
    
    print(f"Average Threshold: {average_threshold}")
    return average_threshold

# Function application
average_threshold_df = calculate_average_threshold(df)
average_threshold_df1 = calculate_average_threshold(df1)



##################################### The methodology is to take the median cosine similarity of the proposed semantic change #########################################


# Load data
df = pd.read_csv('results_pre_trained_incre_OP_300.csv')
df1 = pd.read_csv('results_pre_trained_incre_300.csv')

def calculate_median_threshold(df):
    """
    Calculate the median of the 'Cosine Similarity' column as the threshold.
    """
    # Calculate the median cosine similarity
    median_threshold = df['Cosine Similarity'].median()
    
    print(f"Median Threshold: {median_threshold}")
    return median_threshold

# Function application
median_threshold_df = calculate_median_threshold(df)
median_threshold_df1 = calculate_median_threshold(df1)





##################################### The methodology is to take the avg - confidence interval of cosine similarity of the proposed semantic change #########################################


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

def calculate_threshold_and_update_classification(df, method="average", confidence_level=0.95, top_percentage=5):
    """
    Calculate the threshold based on the specified method, update the 'Classification' column
    based on the threshold, and return a new DataFrame and accuracy.
    
    Parameters:
    - df (DataFrame): The DataFrame containing 'Cosine Similarity', 'Classification', and 'Ground Truth' columns.
    - method (str): The method for calculating the threshold ('median', 'average', 'average - confidence interval', or 'top percentage').
    - confidence_level (float): The confidence level for the confidence interval (default is 0.95).
    - top_percentage (int): The top percentage of lowest values to use for the 'top percentage' threshold method (default is 5).

    Returns:
    - DataFrame: A new DataFrame with the updated 'Classification' column.
    - float: The accuracy based on the updated 'Classification' and 'Ground Truth' columns.
    - float: The threshold used.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure necessary columns are present
    if 'Cosine Similarity' not in df_copy.columns or 'Ground Truth' not in df_copy.columns:
        raise ValueError("DataFrame must contain 'Cosine Similarity' and 'Ground Truth' columns.")
    
    # Calculate threshold based on the specified method
    if method == "median":
        threshold = df_copy['Cosine Similarity'].median()
    elif method == "average":
        threshold = df_copy['Cosine Similarity'].mean()
    elif method == "average - confidence interval":
        # Calculate mean and standard deviation
        mean_similarity = df_copy['Cosine Similarity'].mean()
        std_similarity = df_copy['Cosine Similarity'].std()
        n = len(df_copy['Cosine Similarity'])
        
        # Z-score for the specified confidence level
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Calculate confidence interval
        confidence_interval = z * (std_similarity / np.sqrt(n))
        
        # Calculate threshold as mean minus confidence interval
        threshold = mean_similarity - confidence_interval
    elif method == "top percentage":
        # Calculate threshold as the top 'top_percentage' percentile of lowest values
        threshold = np.percentile(df_copy['Cosine Similarity'], top_percentage)
    else:
        raise ValueError("Invalid method specified. Choose 'median', 'average', 'average - confidence interval', or 'top percentage'.")
    
    # Print current classifications for comparison
    print("Original Classification sample (before update):")
    print(df_copy['Classification'].head(10))

    # Update 'Classification' in the copy based on the calculated threshold
    df_copy['Classification'] = (df_copy['Cosine Similarity'] < threshold).astype(int)
    
    # Print updated classifications for verification
    print("Updated Classification sample (after update):")
    print(df_copy['Classification'].head(10))
    
    # Calculate accuracy by comparing updated 'Classification' with 'Ground Truth'
    accuracy = accuracy_score(df_copy['Ground Truth'], df_copy['Classification'])
    
    print(f"Threshold ({method}): {threshold}")
    print(f"Accuracy based on threshold: {accuracy}")
    
    return df_copy, accuracy, threshold





# 1)  Result of Pre trained incre OP
df = pd.read_csv('results_pre_trained_incre_OP_300.csv')

# Using median as threshold method
updated_df_median, accuracy_median, threshold = calculate_threshold_and_update_classification(df, method="median")

# Using average as threshold method
updated_df_average, accuracy_average, threshold  = calculate_threshold_and_update_classification(df, method="average")

# Using average - confidence interval as threshold method
updated_df_avg_ci, accuracy_avg_ci, threshold  = calculate_threshold_and_update_classification(df, method="average - confidence interval",confidence_level=0.9)



# 2)  Result of Pre trained incre OP
df = pd.read_csv('results_pre_trained_incre_300.csv')

# Using median as threshold method
updated_df_median, accuracy_median, threshold  = calculate_threshold_and_update_classification(df, method="median")

# Using average as threshold method
updated_df_average, accuracy_average, threshold  = calculate_threshold_and_update_classification(df, method="average")

# Using average - confidence interval as threshold method
updated_df_avg_ci, accuracy_avg_ci, threshold  = calculate_threshold_and_update_classification(df, method="average - confidence interval",confidence_level=0.9)

# Using top 5 as threshold method
updated_df, accuracy, threshold = calculate_threshold_and_update_classification(df, method="top percentage", top_percentage=5)




# 3)  Result of Pre trained incre OP
df = pd.read_csv('results_pre_trained_200.csv')

# Using median as threshold method
updated_df_median, accuracy_median, threshold  = calculate_threshold_and_update_classification(df, method="median")

# Using average as threshold method
updated_df_average, accuracy_average, threshold  = calculate_threshold_and_update_classification(df, method="average")

# Using average - confidence interval as threshold method
updated_df_avg_ci, accuracy_avg_ci, threshold  = calculate_threshold_and_update_classification(df, method="average - confidence interval",confidence_level=0.9)


# Using top 5 as threshold method
updated_df, accuracy, threshold = calculate_threshold_and_update_classification(df, method="top percentage", top_percentage=5)




# 4)  Result of Pre trained incre OP
df = pd.read_csv('results_trained_200.csv')

# Using median as threshold method
updated_df_median, accuracy_median, threshold  = calculate_threshold_and_update_classification(df, method="median")

# Using average as threshold method
updated_df_average, accuracy_average, threshold  = calculate_threshold_and_update_classification(df, method="average")

# Using average - confidence interval as threshold method
updated_df_avg_ci, accuracy_avg_ci, threshold  = calculate_threshold_and_update_classification(df, method="average - confidence interval",confidence_level=0.90)

# Using top 5 as threshold method
updated_df, accuracy, threshold = calculate_threshold_and_update_classification(df, method="top percentage", top_percentage=5)






###############################################################  Final Threshold  ######################################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec

def calculate_thresholds_and_plot(model_1, model_2, word_type="nn", alignment="OP", confidence_level=0.95, extreme_percentile=5):
    """
    Calculate thresholds based on cosine similarity for common words between two models and plot the distribution.

    Parameters:
    - model_1 (Word2Vec): The first Word2Vec model.
    - model_2 (Word2Vec): The second Word2Vec model.
    - word_type (str): Type of word to filter ("nn" for nouns or "vb" for verbs).
    - alignment (str): Type of alignment ("OP" for Orthogonal Procrustes, "incremental" for incremental).
    - confidence_level (float): Confidence level for the confidence interval.
    - extreme_percentile (int): Percentile threshold for extreme values (default is 5 for the 5th percentile).

    Returns:
    - dict: A dictionary containing the calculated thresholds for median, average, mean - confidence interval, and extreme percentile.
    """
    
    # Find common words with specified word type
    common_vocab = [word for word in model_1.wv.index_to_key if word in model_2.wv.index_to_key and word.endswith(f"_{word_type}")]
    print(f"Number of common words: {len(common_vocab)}")
    
    # Extract embeddings for common words
    embeddings1 = np.array([model_1.wv[word] for word in common_vocab])
    embeddings2 = np.array([model_2.wv[word] for word in common_vocab])

    # Apply alignment if specified
    if alignment == "OP":
        # Orthogonal Procrustes alignment
        R, _ = orthogonal_procrustes(embeddings1, embeddings2)
        aligned_embeddings1 = embeddings1.dot(R)
    elif alignment == "incremental":
        # No explicit alignment; use embeddings directly
        aligned_embeddings1 = embeddings1
    else:
        raise ValueError("Invalid alignment type. Choose 'OP' or 'incremental'.")

    # Calculate cosine similarities using vectorized operations
    norms1 = np.linalg.norm(aligned_embeddings1, axis=1)
    norms2 = np.linalg.norm(embeddings2, axis=1)
    dot_products = np.einsum('ij,ij->i', aligned_embeddings1, embeddings2)
    cosine_similarities = dot_products / (norms1 * norms2)
    
    # Remove any NaN values from cosine similarities
    cosine_similarities = cosine_similarities[~np.isnan(cosine_similarities)]
    
    # Calculate the thresholds
    thresholds = {
        "median": np.median(cosine_similarities),
        "average": np.mean(cosine_similarities)
    }

    # Mean minus confidence interval threshold
    mean_similarity = thresholds["average"]
    std_similarity = np.std(cosine_similarities)
    n = len(cosine_similarities)
    
    # Z-score for the specified confidence level
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Calculate confidence interval for the mean
    confidence_interval = z * (std_similarity / np.sqrt(n))
    thresholds["mean - confidence interval"] = mean_similarity - confidence_interval

    # Add threshold based on extreme percentile
    thresholds[f"{extreme_percentile}th percentile"] = np.percentile(cosine_similarities, extreme_percentile)

    # Print thresholds
    for method, value in thresholds.items():
        print(f"Threshold ({method}): {value}")

    # Plot the distribution of cosine similarities with threshold lines
    plt.figure(figsize=(10, 6))
    plt.hist(cosine_similarities, bins=30, edgecolor='black', alpha=0.7, label="Cosine Similarities")
    
    # Define colors for each threshold line
    threshold_colors = {
        "median": "blue",
        "average": "green",
        "mean - confidence interval": "red",
        f"{extreme_percentile}th percentile": "purple"
    }
    
    # Plot each threshold with its specific color
    for method, value in thresholds.items():
        plt.axvline(x=value, color=threshold_colors[method], linestyle='--', label=f'{method} ({value:.3f})')

    # Plot settings
    plt.title(f"Distribution of Cosine Similarities ({word_type}, {alignment} alignment)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    return thresholds

# Example usage
model_1 = Word2Vec.load('fine_tuned_model_1_epochs_300.model')
model_2 = Word2Vec.load('fine_tuned_incremental_model_1_epochs_300.model')

# Calculate all thresholds for common nouns with Orthogonal Procrustes alignment and plot the distribution
thresholds = calculate_thresholds_and_plot(model_1, model_2, word_type="nn", alignment="OP", confidence_level=0.90, extreme_percentile=5)











################################################################ Propose words for annotation ###############################################################################



def extract_align_and_filter_words(model1, model2, threshold, alignment="OP", count_n=5, top_n=300, pos_filter="both"):
    """
    Extract embeddings for common words that appear at least three times in both models,
    align them using Orthogonal Procrustes if specified, calculate cosine similarity, 
    and return the top words with cosine similarity below the threshold.
    
    Parameters:
    model1: Trained Word2Vec model for corpus1.
    model2: Trained Word2Vec model for corpus2.
    threshold (float): The cosine similarity threshold for identifying potential words for annotation.
    alignment (str): Alignment type, either "OP" for Orthogonal Procrustes or "no_alignment" for incremental learning.
    top_n (int): Number of top words to return, ordered by lowest cosine similarity.
    pos_filter (str): Part-of-speech filter, options are "nn" for nouns, "vb" for verbs, or "both" for all.
    
    Returns:
    list: A list of the top words with cosine similarity below the threshold (proposed for annotation).
    dict: A dictionary with top_n proposed words and their cosine similarities.
    """
    
    # Find common words between the two models
    common_words = list(set(model1.wv.index_to_key).intersection(set(model2.wv.index_to_key)))

    # Filter words that appear at least 3 times in each corpus and match the pos_filter
    filtered_common_words = [
        word for word in common_words 
        if model1.wv.get_vecattr(word, 'count') >= count_n and model2.wv.get_vecattr(word, 'count') >= count_n
        and len(word) > 6 and not any(char.isdigit() for char in word)
        and (pos_filter == "both" or (pos_filter == "nn" and word.endswith("_nn")) or (pos_filter == "vb" and word.endswith("_vb")))
    ]
    
    if not filtered_common_words:
        raise ValueError("No common words found between the two models after filtering.")
    
    # Extract embeddings for the filtered common words
    embeddings1 = np.array([model1.wv[word] for word in filtered_common_words])
    embeddings2 = np.array([model2.wv[word] for word in filtered_common_words])
    
    # Apply alignment if specified
    if alignment == "OP":
        # Align the embeddings using Orthogonal Procrustes
        A = embeddings1
        B = embeddings2
        M = np.dot(B.T, A)
        u, _, vh = svd(M, full_matrices=False)
        R = np.dot(u, vh)
        aligned_embeddings1 = np.dot(A, R)
    else:
        # No alignment (incremental learning)
        aligned_embeddings1 = embeddings1

    # Calculate cosine similarity and filter words
    proposed_words = []

    for i, word in enumerate(filtered_common_words):
        norm1 = np.linalg.norm(aligned_embeddings1[i])
        norm2 = np.linalg.norm(embeddings2[i])
        
        if norm1 > 0 and norm2 > 0:
            cos_sim = np.dot(aligned_embeddings1[i], embeddings2[i]) / (norm1 * norm2)
            
            # Filter based on the threshold
            if cos_sim < threshold:
                proposed_words.append((word, cos_sim))

    # Sort proposed words by cosine similarity in ascending order and select top_n
    proposed_words = sorted(proposed_words, key=lambda x: x[1])[:top_n]
    top_words = [word for word, _ in proposed_words]
    
    # Create similarity dictionary for only the top_n proposed words
    similarity_dict = {word: cos_sim for word, cos_sim in proposed_words}

    return top_words, similarity_dict



# Testing the function with a threshold and top_n
threshold = 0.887

# Load the actual models instead of assigning strings
model_1 = Word2Vec.load('fine_tuned_model_1_epochs_300.model')
model_2 = Word2Vec.load('fine_tuned_incremental_model_1_epochs_300.model')


top_words, similarity_dict = extract_align_and_filter_words(model_1, model_2, threshold, alignment="Incremental",count_n=100, top_n=40000, pos_filter="nn")

print("Top Words:", top_words)
print("Similarity Dictionary:", similarity_dict)


len(top_words)




def clean_words(words):
    # Use list comprehension to remove '_nn' and '_vb' suffixes from each word
    top_words = [word.replace('_nn', '').replace('_vb', '') for word in words]
    return top_words


top_words_clean = clean_words(top_words)
print(top_words_clean)






###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################



################################################ Create a csv file for the annotators  #####################################################################################################################




################################################## (I) Random approach ################################################################################################

import pandas as pd
import random



# Step 1: Identify Target Words and Preprocess Data
def get_word_occurrences(preprocessed_corpus, selected_words, original_corpus_length):
    """
    Identifies and returns the indexes of sentences containing each target word in the preprocessed corpus,
    ensuring indexes are within the bounds of the original corpus.
    """
    occurrences = {word: [] for word in selected_words}
    for idx, sentence in enumerate(preprocessed_corpus):
        if idx >= original_corpus_length:
            break  # Stop if index exceeds original corpus length
        for word in selected_words:
            if word in sentence:
                occurrences[word].append(idx)
    return occurrences

# Step 2 & 3: Collect Sentence Occurrences and Perform Random Sampling
def sample_representative_sentences(occurrences, original_corpus, n):
    """
    Randomly samples up to `n` representative sentences for each word from the occurrences.
    If fewer than `n` occurrences exist, all occurrences are returned.
    """
    sampled_sentences = {}
    for word, indexes in occurrences.items():
        # Randomly sample n indexes if there are more than n occurrences
        if len(indexes) > n:
            selected_indexes = random.sample(indexes, n)
        else:
            selected_indexes = indexes
        
        # Retrieve sampled sentences and join with '|'
        sentences = [original_corpus[idx] for idx in selected_indexes if idx < len(original_corpus)]
        sampled_sentences[word] = '        ||        '.join(sentences)
    return sampled_sentences

# Step 4: Generate the Data for Excel Output
def generate_excel_for_annotation(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing representative sentences for annotation.
    """
    data = []

    # Process corpus1 and corpus2 and gather data for each word
    occurrences_corpus1 = get_word_occurrences(preprocessed_corpus1, selected_words, len(original_corpus1))
    sampled_corpus1 = sample_representative_sentences(occurrences_corpus1, original_corpus1, n)
    
    occurrences_corpus2 = get_word_occurrences(preprocessed_corpus2, selected_words, len(original_corpus2))
    sampled_corpus2 = sample_representative_sentences(occurrences_corpus2, original_corpus2, n)

    # Combine data into a single structure for Excel output
    for word in selected_words:
        data.append({
            'word': word,
            'corpus1_sentences': sampled_corpus1.get(word, ''),
            'corpus2_sentences': sampled_corpus2.get(word, ''),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'random_representative_sentences_for_annotation.xlsx'.")



# Execute the pipeline

generate_excel_for_annotation(top_words, corpus_1_pos, doc1, corpus_2_pos, doc2, 5)




################################################## (II) Diversity based approach ################################################################################################

"""
Diversity-Based Sampling with Sentence Embeddings


1. Diversity-Based Sampling
Goal: Select sentences that capture a broad range of contexts in which the word appears.
Approach: Cluster sentences containing the word (using sentence embeddings) and then sample one sentence per cluster, up to n sentences.
This approach ensures that annotators see various contexts rather than highly similar examples.

2. Contextual Similarity Scoring
Goal: Select sentences most representative of each corpus's typical context for the word.
Approach: Compute a similarity score (using cosine similarity) between the embedding of each sentence and a prototypical embedding of the word's context (such as the average embedding across all sentences containing the word). Select the top n highest-scoring sentences.
This approach selects sentences that best represent the general usage of the word in each corpus.

3. Semantic Change Relevance Scoring
Goal: Select sentences that may be indicative of semantic change.
Approach: Calculate the average embedding for sentences containing the word in both corpora and select sentences whose embeddings diverge the most from the average of the opposite corpus.


"""

""" 
Explanation of the Sophisticated Pipeline
Sentence Embeddings: For each word, we embed the sentences that contain the word in each corpus.
Clustering: We use KMeans clustering to group the embeddings. The number of clusters is set to the minimum of n or the total sentences available, ensuring that we don’t create more clusters than sentences.
Selecting Representative Sentences: For each cluster, the sentence closest to the cluster center is selected, capturing a broad range of contexts.
Excel Output: Finally, the diverse representative sentences are saved in the requested format.

"""



import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models based on accuracy/speed tradeoff

def get_sentence_embeddings(sentences):
    """Generates sentence embeddings using a pre-trained transformer model."""
    return model.encode(sentences)

def select_diverse_sentences(sentences, n):
    """Selects diverse sentences by clustering and sampling one sentence per cluster."""
    embeddings = get_sentence_embeddings(sentences)
    
    # Determine the number of clusters as the minimum of n and number of sentences
    num_clusters = min(n, len(sentences))
    
    # Cluster the sentence embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    # For each cluster, find the closest sentence to the cluster center
    selected_sentences = []
    for i in range(num_clusters):
        cluster_indices = [j for j, label in enumerate(kmeans.labels_) if label == i]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Find the sentence closest to the cluster center
        closest_idx = cluster_indices[cosine_similarity([cluster_centers[i]], cluster_embeddings).argmax()]
        selected_sentences.append(sentences[closest_idx])
    
    return selected_sentences

# Sample sophisticated approach within the pipeline
def sophisticated_generate_excel_for_annotation(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing diverse representative sentences for annotation using clustering.
    """
    data = []

    for word in selected_words:
        # Filter sentences containing the word for both corpora
        sentences_corpus1 = [original_corpus1[idx] for idx, sentence in enumerate(preprocessed_corpus1) if word in sentence]
        sentences_corpus2 = [original_corpus2[idx] for idx, sentence in enumerate(preprocessed_corpus2) if word in sentence]
        
        # Select diverse sentences using clustering-based approach
        selected_sentences_corpus1 = select_diverse_sentences(sentences_corpus1, n)
        selected_sentences_corpus2 = select_diverse_sentences(sentences_corpus2, n)
        
        # Join sentences with '|' and add to the final data structure
        data.append({
            'word': word,
            'corpus1_sentences': '        ||        '.join(selected_sentences_corpus1),
            'corpus2_sentences': '        ||        '.join(selected_sentences_corpus2),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('sophisticated_representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'sophisticated_representative_sentences_for_annotation.xlsx'.")





# Execute the sophisticated pipeline
sophisticated_generate_excel_for_annotation(top_words, corpus_1_pos, doc1, corpus_2_pos, doc2, 5)




################################################## (III) Contextual Similarity Scoring approach  ################################################################################################

"""

The Contextual Similarity Scoring approach focuses on selecting sentences that are most representative of the typical context in which each word appears in each corpus. By scoring sentences based on their similarity to an "average" or "prototypical" context for the word, we can highlight sentences that best encapsulate its general usage within each corpus.


Explanation of Code
Prototypical Embedding Calculation:

For each target word, the calculate_prototypical_embedding function calculates the mean embedding of all sentences containing the word in each corpus. This average serves as a "prototypical" embedding that reflects the typical context of the word.
Similarity Scoring and Selection:

Each sentence’s embedding is compared to the prototypical embedding using cosine similarity. The top n sentences with the highest similarity scores are selected as the most representative.
Excel Output:

The selected sentences for each word in each corpus are joined with | and saved in an Excel file, along with an empty annotation column for annotators to fill.
Benefits of This Approach
Captures Typical Context: By selecting sentences closest to the prototypical context, annotators are provided with examples that best represent the average use of each word in each corpus.
Avoids Outliers: This method filters out atypical usages, helping annotators see the most contextually relevant sentences.


"""





import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models based on accuracy/speed tradeoff

def get_sentence_embeddings(sentences):
    """Generates sentence embeddings using a pre-trained transformer model."""
    return model.encode(sentences)

def calculate_prototypical_embedding(sentences):
    """Calculates the prototypical embedding by averaging sentence embeddings."""
    embeddings = get_sentence_embeddings(sentences)
    prototypical_embedding = np.mean(embeddings, axis=0)
    return prototypical_embedding

def select_top_similar_sentences(sentences, prototypical_embedding, n):
    """Selects top n sentences with highest similarity to the prototypical embedding."""
    embeddings = get_sentence_embeddings(sentences)
    similarities = cosine_similarity([prototypical_embedding], embeddings).flatten()
    top_n_indices = similarities.argsort()[-n:][::-1]  # Get indices of top n similar sentences
    return [sentences[idx] for idx in top_n_indices]

# Sample sophisticated approach within the pipeline
def contextual_similarity_generate_excel_for_annotation(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing representative sentences based on contextual similarity scoring for annotation.
    """
    data = []

    for word in selected_words:
        # Filter sentences containing the word for both corpora
        sentences_corpus1 = [original_corpus1[idx] for idx, sentence in enumerate(preprocessed_corpus1) if word in sentence]
        sentences_corpus2 = [original_corpus2[idx] for idx, sentence in enumerate(preprocessed_corpus2) if word in sentence]
        
        # Calculate prototypical embeddings
        if sentences_corpus1:
            prototypical_embedding_corpus1 = calculate_prototypical_embedding(sentences_corpus1)
            selected_sentences_corpus1 = select_top_similar_sentences(sentences_corpus1, prototypical_embedding_corpus1, n)
        else:
            selected_sentences_corpus1 = []

        if sentences_corpus2:
            prototypical_embedding_corpus2 = calculate_prototypical_embedding(sentences_corpus2)
            selected_sentences_corpus2 = select_top_similar_sentences(sentences_corpus2, prototypical_embedding_corpus2, n)
        else:
            selected_sentences_corpus2 = []

        # Join sentences with '|' and add to the final data structure
        data.append({
            'word': word,
            'corpus1_sentences': '        ||        '.join(selected_sentences_corpus1),
            'corpus2_sentences': '        ||        '.join(selected_sentences_corpus2),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('contextual_similarity_representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'contextual_similarity_representative_sentences_for_annotation.xlsx'.")




# Execute the sophisticated pipeline

contextual_similarity_generate_excel_for_annotation(top_words, corpus_1_pos, doc1, corpus_2_pos, doc2, 5)





################################################## (IV) Semantic Change Relevance Scoring approach  ################################################################################################


'''
The Semantic Change Relevance Scoring approach aims to capture sentences that highlight potential semantic changes by selecting sentences that exhibit a marked difference in context between two corpora. This can help annotators identify shifts in meaning by showing sentences that diverge the most from the typical usage of each word in the other corpus.

Approach Outline
Compute Prototypical Embeddings for Each Word in Each Corpus:

Generate embeddings for sentences containing the target word in each corpus.
Calculate the average embedding for each word in each corpus, which serves as the “prototypical context.”
Score Sentences by Cross-Corpus Divergence:

For each sentence in one corpus, calculate the cosine similarity to the prototypical embedding from the other corpus.
Lower similarity scores indicate contexts that are less typical according to the other corpus, suggesting potential semantic divergence.
Select Sentences with Lowest Cross-Corpus Similarity:

Sort the sentences by their cross-corpus similarity score and select the n lowest-scoring sentences for each corpus.

'''


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models based on accuracy/speed tradeoff

def get_sentence_embeddings(sentences):
    """Generates sentence embeddings using a pre-trained transformer model."""
    return model.encode(sentences)

def calculate_prototypical_embedding(sentences):
    """Calculates the prototypical embedding by averaging sentence embeddings."""
    embeddings = get_sentence_embeddings(sentences)
    prototypical_embedding = np.mean(embeddings, axis=0)
    return prototypical_embedding

def select_lowest_similarity_sentences(sentences, opposite_prototypical_embedding, n):
    """Selects top n sentences with lowest similarity to the prototypical embedding from the opposite corpus."""
    embeddings = get_sentence_embeddings(sentences)
    similarities = cosine_similarity([opposite_prototypical_embedding], embeddings).flatten()
    lowest_n_indices = similarities.argsort()[:n]  # Get indices of lowest n similar sentences
    return [sentences[idx] for idx in lowest_n_indices]

# Sample sophisticated approach within the pipeline
def semantic_change_relevance_generate_excel_for_annotation(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing representative sentences based on semantic change relevance scoring for annotation.
    """
    data = []

    for word in selected_words:
        # Filter sentences containing the word for both corpora
        sentences_corpus1 = [original_corpus1[idx] for idx, sentence in enumerate(preprocessed_corpus1) if word in sentence]
        sentences_corpus2 = [original_corpus2[idx] for idx, sentence in enumerate(preprocessed_corpus2) if word in sentence]
        
        # Calculate prototypical embeddings for cross-corpus scoring
        if sentences_corpus1:
            prototypical_embedding_corpus1 = calculate_prototypical_embedding(sentences_corpus1)
        if sentences_corpus2:
            prototypical_embedding_corpus2 = calculate_prototypical_embedding(sentences_corpus2)

        # Select sentences with lowest similarity to the opposite corpus's prototypical embedding
        if sentences_corpus1 and sentences_corpus2:
            selected_sentences_corpus1 = select_lowest_similarity_sentences(sentences_corpus1, prototypical_embedding_corpus2, n)
            selected_sentences_corpus2 = select_lowest_similarity_sentences(sentences_corpus2, prototypical_embedding_corpus1, n)
        else:
            selected_sentences_corpus1 = sentences_corpus1[:n] if sentences_corpus1 else []
            selected_sentences_corpus2 = sentences_corpus2[:n] if sentences_corpus2 else []

        # Join sentences with '|' and add to the final data structure
        data.append({
            'word': word,
            'corpus1_sentences': '        ||        '.join(selected_sentences_corpus1),
            'corpus2_sentences': '        ||        '.join(selected_sentences_corpus2),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('semantic_change_relevance_representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'semantic_change_relevance_representative_sentences_for_annotation.xlsx'.")



# Execute the sophisticated pipeline

semantic_change_relevance_generate_excel_for_annotation(top_words, corpus_1_pos, doc1_clean, corpus_2_pos, doc2_clean, 5)





################################################## (V) Semantic Change Relevance Scoring approach with own models  ################################################################################################

'''
This approach ensures that the embeddings are fine-tuned to capture the specific characteristics of each corpus and will likely yield more accurate representations of semantic change.

'''



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np

# Paths to your saved models for corpus1 and corpus2
MODEL_PATH_CORPUS1 = "fine_tuned_model_1_epochs_300.model"
MODEL_PATH_CORPUS2 = "fine_tuned_incremental_model_1_epochs_300.model"

# Load the models using Gensim's Word2Vec format
model_corpus1 = Word2Vec.load(MODEL_PATH_CORPUS1)
model_corpus2 = Word2Vec.load(MODEL_PATH_CORPUS2)

# Define functions to get sentence embeddings by averaging word embeddings
def get_sentence_embedding(sentence, model):
    """Generate a sentence embedding by averaging word embeddings from the model."""
    words = [word for word in sentence if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words], axis=0)

def calculate_prototypical_embedding(sentences, model):
    """Calculates the prototypical embedding by averaging sentence embeddings."""
    embeddings = [get_sentence_embedding(sentence, model) for sentence in sentences]
    prototypical_embedding = np.mean(embeddings, axis=0)
    return prototypical_embedding

def select_lowest_similarity_sentences(sentences, opposite_prototypical_embedding, model, n):
    """Selects top n sentences with lowest similarity to the prototypical embedding from the opposite corpus."""
    embeddings = [get_sentence_embedding(sentence, model) for sentence in sentences]
    similarities = cosine_similarity([opposite_prototypical_embedding], embeddings).flatten()
    lowest_n_indices = similarities.argsort()[:n]  # Get indices of lowest n similar sentences
    return [sentences[idx] for idx in lowest_n_indices]

# Main function to generate the Excel for annotation
def semantic_change_own_relevance_generate_excel_for_annotation(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing representative sentences based on semantic change relevance scoring for annotation.
    """
    data = []

    for word in selected_words:
        # Filter sentences containing the word for both corpora
        sentences_corpus1 = [original_corpus1[idx] for idx, sentence in enumerate(preprocessed_corpus1) if word in sentence]
        sentences_corpus2 = [original_corpus2[idx] for idx, sentence in enumerate(preprocessed_corpus2) if word in sentence]
        
        # Calculate prototypical embeddings for cross-corpus scoring
        if sentences_corpus1:
            prototypical_embedding_corpus1 = calculate_prototypical_embedding(sentences_corpus1, model_corpus1)
        if sentences_corpus2:
            prototypical_embedding_corpus2 = calculate_prototypical_embedding(sentences_corpus2, model_corpus2)

        # Select sentences with lowest similarity to the opposite corpus's prototypical embedding
        if sentences_corpus1 and sentences_corpus2:
            selected_sentences_corpus1 = select_lowest_similarity_sentences(sentences_corpus1, prototypical_embedding_corpus2, model_corpus1, n)
            selected_sentences_corpus2 = select_lowest_similarity_sentences(sentences_corpus2, prototypical_embedding_corpus1, model_corpus2, n)
        else:
            selected_sentences_corpus1 = sentences_corpus1[:n] if sentences_corpus1 else []
            selected_sentences_corpus2 = sentences_corpus2[:n] if sentences_corpus2 else []

        # Join sentences with '|' and add to the final data structure
        data.append({
            'word': word,
            'corpus1_sentences': '        ||        '.join(selected_sentences_corpus1),
            'corpus2_sentences': '        ||        '.join(selected_sentences_corpus2),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('semantic_change_own_relevance_representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'semantic_change_own_relevance_representative_sentences_for_annotation.xlsx'.")



 
# Execute the sophisticated pipeline with saved models
semantic_change_own_relevance_generate_excel_for_annotation(top_words, corpus_1_pos, doc1_clean, corpus_2_pos, doc2_clean, 5)






################################################## (VI) Semantic Change Relevance Scoring approach with own models  ################################################################################################






import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np

# Paths to your saved models for corpus1 and corpus2
MODEL_PATH_CORPUS1 = "fine_tuned_model_1_epochs_200.model"
MODEL_PATH_CORPUS2 = "fine_tuned_incremental_model_1_epochs_200.model"

# Load the models using Gensim's Word2Vec format
model_corpus1 = Word2Vec.load(MODEL_PATH_CORPUS1)
model_corpus2 = Word2Vec.load(MODEL_PATH_CORPUS2)

# Define functions to get sentence embeddings by averaging word embeddings
def get_sentence_embedding(sentence, model):
    """Generate a sentence embedding by averaging word embeddings from the model."""
    words = [word for word in sentence if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words], axis=0)

def calculate_prototypical_embedding(sentences, model):
    """Calculates the prototypical embedding by averaging sentence embeddings."""
    embeddings = [get_sentence_embedding(sentence, model) for sentence in sentences]
    prototypical_embedding = np.mean(embeddings, axis=0)
    return prototypical_embedding

def select_top_similar_sentences(sentences, prototypical_embedding, model, n):
    """Selects top n sentences with highest similarity to the prototypical embedding."""
    embeddings = [get_sentence_embedding(sentence, model) for sentence in sentences]
    similarities = cosine_similarity([prototypical_embedding], embeddings).flatten()
    top_n_indices = similarities.argsort()[-n:][::-1]  # Get indices of top n similar sentences
    return [sentences[idx] for idx in top_n_indices]

# Main function to generate the Excel for annotation
def contextual_similarity_own_generate_excel_for_annotation(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing representative sentences based on contextual similarity scoring for annotation.
    """
    data = []

    for word in selected_words:
        # Filter sentences containing the word for both corpora
        sentences_corpus1 = [original_corpus1[idx] for idx, sentence in enumerate(preprocessed_corpus1) if word in sentence]
        sentences_corpus2 = [original_corpus2[idx] for idx, sentence in enumerate(preprocessed_corpus2) if word in sentence]
        
        # Calculate prototypical embeddings for similarity scoring
        if sentences_corpus1:
            prototypical_embedding_corpus1 = calculate_prototypical_embedding(sentences_corpus1, model_corpus1)
            selected_sentences_corpus1 = select_top_similar_sentences(sentences_corpus1, prototypical_embedding_corpus1, model_corpus1, n)
        else:
            selected_sentences_corpus1 = []

        if sentences_corpus2:
            prototypical_embedding_corpus2 = calculate_prototypical_embedding(sentences_corpus2, model_corpus2)
            selected_sentences_corpus2 = select_top_similar_sentences(sentences_corpus2, prototypical_embedding_corpus2, model_corpus2, n)
        else:
            selected_sentences_corpus2 = []

        # Join sentences with '|' and add to the final data structure
        data.append({
            'word': word,
            'corpus1_sentences': '        ||        '.join(selected_sentences_corpus1),
            'corpus2_sentences': '        ||        '.join(selected_sentences_corpus2),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('contextual_similarity_own_representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'contextual_similarity_own_representative_sentences_for_annotation.xlsx'.")



# Execute the sophisticated pipeline with saved models
contextual_similarity_own_generate_excel_for_annotation(top_words, corpus_1_pos, doc1_clean, corpus_2_pos, doc2_clean, 5)








################################################## (VII) K-Means approach with own models  ################################################################################################


import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np

# Paths to your saved models for corpus1 and corpus2
MODEL_PATH_CORPUS1 = "fine_tuned_model_1_epochs_200.model"
MODEL_PATH_CORPUS2 = "fine_tuned_incremental_model_1_epochs_200.model"

# Load the models using Gensim's Word2Vec format
model_corpus1 = Word2Vec.load(MODEL_PATH_CORPUS1)
model_corpus2 = Word2Vec.load(MODEL_PATH_CORPUS2)

# Define functions to get sentence embeddings by averaging word embeddings
def get_sentence_embedding(sentence, model):
    """Generate a sentence embedding by averaging word embeddings from the model."""
    words = [word for word in sentence if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words], axis=0)

def select_diverse_sentences(sentences, model, n):
    """
    Use K-means clustering to select n diverse representative sentences.
    
    Parameters:
    - sentences: List of sentences to cluster.
    - model: Word2Vec model to generate embeddings.
    - n: Number of clusters/representative sentences to select.

    Returns:
    - List of representative sentences, one from each cluster.
    """
    # Calculate embeddings for each sentence
    embeddings = np.array([get_sentence_embedding(sentence, model) for sentence in sentences])

    # If there are fewer sentences than clusters, return all sentences
    if len(sentences) <= n:
        return sentences

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(embeddings)
    
    # Select the sentence closest to each cluster center
    representative_sentences = []
    for cluster_idx in range(n):
        cluster_center = kmeans.cluster_centers_[cluster_idx]
        cluster_distances = np.linalg.norm(embeddings - cluster_center, axis=1)
        closest_sentence_idx = cluster_distances.argmin()
        representative_sentences.append(sentences[closest_sentence_idx])
    
    return representative_sentences

# Main function to generate the Excel for annotation
def generate_excel_for_annotation_with_kmeans(selected_words, preprocessed_corpus1, original_corpus1, preprocessed_corpus2, original_corpus2, n):
    """
    Generates an Excel file containing representative sentences using K-means clustering for annotation.
    """
    data = []

    for word in selected_words:
        # Filter sentences containing the word for both corpora
        sentences_corpus1 = [original_corpus1[idx] for idx, sentence in enumerate(preprocessed_corpus1) if word in sentence]
        sentences_corpus2 = [original_corpus2[idx] for idx, sentence in enumerate(preprocessed_corpus2) if word in sentence]
        
        # Use K-means clustering to select diverse representative sentences
        if sentences_corpus1:
            selected_sentences_corpus1 = select_diverse_sentences(sentences_corpus1, model_corpus1, n)
        else:
            selected_sentences_corpus1 = []

        if sentences_corpus2:
            selected_sentences_corpus2 = select_diverse_sentences(sentences_corpus2, model_corpus2, n)
        else:
            selected_sentences_corpus2 = []

        # Join sentences with '||' and add to the final data structure
        data.append({
            'word': word,
            'corpus1_sentences': '        ||        '.join(selected_sentences_corpus1),
            'corpus2_sentences': '        ||        '.join(selected_sentences_corpus2),
            'annotation': ''  # Empty column for annotation
        })

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'])
    df.to_excel('kmeans_representative_sentences_for_annotation.xlsx', index=False)
    print("Excel file created as 'kmeans_representative_sentences_for_annotation.xlsx'.")

# Execute the pipeline with saved models
generate_excel_for_annotation_with_kmeans(top_words, corpus_1_pos, doc1_clean, corpus_2_pos, doc2_clean, 5)









########################################################################################## Finalize the preprocessed dataset - Majority vote ########################################################################################

import pandas as pd
from collections import Counter

def majority_vote_annotation_xlsx_to_csv(xlsx_files, output_file="final_annotation.csv"):
    """
    Create a final CSV file with words and majority-voted annotations from multiple annotation Excel files.
    
    Parameters:
    xlsx_files (list): List of Excel file names (paths) to be merged.
    output_file (str): Name of the final CSV file with majority-voted annotations.
    
    Returns:
    pd.DataFrame: DataFrame containing words and their majority-voted annotations.
    """
    # Step 1: Load all Excel files into a list of DataFrames
    dfs = [pd.read_excel(file) for file in xlsx_files]
    
    # Ensure all DataFrames have the correct columns
    required_columns = {'word', 'corpus1_sentences', 'corpus2_sentences', 'annotation'}
    for df in dfs:
        if not required_columns.issubset(df.columns):
            raise ValueError("Each Excel file must contain 'word', 'corpus1_sentences', 'corpus2_sentences', and 'annotation' columns.")
    
    # Step 2: Initialize a DataFrame for the final output (based on the first file)
    final_df = dfs[0][['word', 'corpus1_sentences', 'corpus2_sentences']].copy()  # Retain required columns for output
    
    # Step 3: For each word, apply majority voting on annotations
    annotations = []
    for i in range(len(final_df)):
        word_annotations = []
        for df in dfs:
            annotation = df.iloc[i].get('annotation', None)
            if pd.notna(annotation):  # Only add if annotation is not NaN
                word_annotations.append(annotation)
        
        # Apply majority vote using Counter if there are annotations available
        if word_annotations:
            majority_annotation = Counter(word_annotations).most_common(1)[0][0]
        else:
            majority_annotation = None  # Set to None if no annotations are found

        annotations.append(majority_annotation)
    
    # Step 4: Add the final annotations to the DataFrame
    final_df['annotation'] = annotations
    
    # Step 5: Write the final DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    
    print(f"Final annotated CSV file '{output_file}' created successfully!")
    
    return final_df


# Example usage:
xlsx_files = ["annotation_file1.xlsx", "annotation_file2.xlsx", "annotation_file3.xlsx"]
final_df = majority_vote_annotation_xlsx_to_csv(xlsx_files, output_file="final_annotation.csv")




##############################################################################################################################################################################################################################################################################

###########################################################   Remove common words from final annotation dataset   ######################################################################################################################

finaldf = pd.read_csv("final_annotation.csv")

finaldf.shape

targets



def remove_rows_with_targets(df, targets):
    """
    Removes rows from the DataFrame where the 'word' column contains values in the target list.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    targets (list): List of target words to remove from the DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Filter rows where 'word' is not in the targets list
    filtered_df = df[~df['word'].isin(targets)].reset_index(drop=True)
    return filtered_df



# Apply the function
filtered_df = remove_rows_with_targets(finaldf, targets)

# Save the filtered DataFrame to a CSV file
filtered_df.to_csv("final_annotation_filtered.csv", index=False)





##############################################################################################################################################################################################################################################################################


########################################################################################## Feature creation for the dataset ##########################################################################################





import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

def create_features(annotated_csv, model_path1, model_path2, corpus1_tokenized, corpus2_tokenized, output_file="features.csv", align_embeddings=False):
    """
    Create a CSV file with features for supervised learning from annotated data.
    
    Parameters:
    annotated_csv (str): Path to the annotated dataset CSV file.
    model_path1 (str): Path to the fine-tuned embeddings model for corpus 1.
    model_path2 (str): Path to the fine-tuned embeddings model for corpus 2.
    corpus1_tokenized (list of list): List of tokenized sentences (each sentence is a list of words) in corpus 1 for TF-IDF calculation.
    corpus2_tokenized (list of list): List of tokenized sentences in corpus 2 for TF-IDF calculation.
    output_file (str): Path for the output feature CSV file.
    align_embeddings (bool): Whether to apply orthogonal Procrustes alignment to the embeddings.
    
    Returns:
    pd.DataFrame: DataFrame with the features for supervised learning.
    """
    # Load annotated data
    annotated_df = pd.read_csv(annotated_csv)
    
    # Load embeddings models
    embeddings1 = Word2Vec.load(model_path1)
    embeddings2 = Word2Vec.load(model_path2)
    
    # Optionally align embeddings using orthogonal Procrustes
    if align_embeddings:
        # Find common vocabulary
        common_vocab = list(set(embeddings1.wv.index_to_key).intersection(set(embeddings2.wv.index_to_key)))
        
        # Extract embeddings for common vocabulary
        matrix1 = np.array([embeddings1.wv[word] for word in common_vocab])
        matrix2 = np.array([embeddings2.wv[word] for word in common_vocab])
        
        # Perform orthogonal Procrustes alignment
        R, _ = orthogonal_procrustes(matrix1, matrix2)
        
        # Align corpus 1 embeddings to corpus 2
        embeddings1_aligned = {word: np.dot(embeddings1.wv[word], R) for word in embeddings1.wv.index_to_key}
    else:
        # No alignment; use the original embeddings
        embeddings1_aligned = {word: embeddings1.wv[word] for word in embeddings1.wv.index_to_key}

    # Convert tokenized sentences to strings for TF-IDF calculation
    corpus1_texts = [" ".join(sentence) for sentence in corpus1_tokenized]
    corpus2_texts = [" ".join(sentence) for sentence in corpus2_tokenized]
    
    # Initialize TF-IDF vectorizers and calculate TF-IDF for each corpus
    vectorizer1 = TfidfVectorizer()
    vectorizer2 = TfidfVectorizer()
    
    tfidf_matrix1 = vectorizer1.fit_transform(corpus1_texts)
    tfidf_matrix2 = vectorizer2.fit_transform(corpus2_texts)
    
    # Map words to TF-IDF scores for both corpora
    tfidf_dict1 = dict(zip(vectorizer1.get_feature_names_out(), tfidf_matrix1.max(axis=0).toarray().flatten()))
    tfidf_dict2 = dict(zip(vectorizer2.get_feature_names_out(), tfidf_matrix2.max(axis=0).toarray().flatten()))
    
    # Create the feature DataFrame
    features = {
        "word": [],
        "corpus1_embedding": [],
        "corpus2_embedding": [],
        "corpus1_tfidf": [],
        "corpus2_tfidf": [],
        "annotation": []
    }
    
    for _, row in annotated_df.iterrows():
        word = row["word"]
        
        # Get aligned embeddings for corpus 1 if available, else use zeros
        emb1 = embeddings1_aligned.get(word, np.zeros(embeddings1.vector_size))
        # Get original embeddings for corpus 2 if available, else use zeros
        emb2 = embeddings2.wv[word] if word in embeddings2.wv else np.zeros(embeddings2.vector_size)
        
        # Get TF-IDF scores if available, else set to 0
        tfidf1 = tfidf_dict1.get(word, 0)
        tfidf2 = tfidf_dict2.get(word, 0)
        
        # Append data to features dictionary
        features["word"].append(word)
        features["corpus1_embedding"].append(emb1)
        features["corpus2_embedding"].append(emb2)
        features["corpus1_tfidf"].append(tfidf1)
        features["corpus2_tfidf"].append(tfidf2)
        features["annotation"].append(row["annotation"])
    
    # Convert to DataFrame and save as CSV
    feature_df = pd.DataFrame(features)
    
    # Expand embedding columns to individual dimensions
    embedding_dim = embeddings1.vector_size
    feature_df[[f"corpus1_emb_{i}" for i in range(embedding_dim)]] = pd.DataFrame(feature_df["corpus1_embedding"].tolist(), index=feature_df.index)
    feature_df[[f"corpus2_emb_{i}" for i in range(embedding_dim)]] = pd.DataFrame(feature_df["corpus2_embedding"].tolist(), index=feature_df.index)
    
    # Drop the original embedding columns
    feature_df.drop(columns=["corpus1_embedding", "corpus2_embedding"], inplace=True)
    
    # Save to CSV
    feature_df.to_csv(output_file, index=False)
    print(f"Feature CSV file '{output_file}' created successfully!")
    
    return feature_df







# Applications 

#1
annotated_csv = "final_annotation_filtered.csv"

model1 = "fine_tuned_model_1_epochs_300.model"
model2 = "fine_tuned_incremental_model_1_epochs_300.model"


create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features.csv",align_embeddings=True)


#2 
annotated_csv = "final_annotation_filtered.csv"

model1 = "fine_tuned_model_1_epochs_300.model"
model2 = "fine_tuned_incremental_model_1_epochs_300.model"


create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features2.csv",align_embeddings=False)


#3
annotated_csv = "final_annotation_filtered.csv"

model1 = "model_1_no_pretrained_epochs_300.model"
model2 = "model_2_no_pretrained_epochs_300.model"


create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features3.csv",align_embeddings=True)


#4
annotated_csv = "final_annotation_filtered.csv"

model1 = "model_1_no_pretrained_epochs_300.model"
model2 = "model_1_incremental_epochs_300.model"


create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features4.csv",align_embeddings=False)


#5
annotated_csv = "final_annotation_filtered.csv"

model1 = "model_1_no_pretrained_epochs_300.model"
model2 = "model_1_incremental_epochs_300.model"


create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features5.csv",align_embeddings=True)






#6 With all words 2+3+4 
annotated_csv = "final_annotation.csv"

model1 = "fine_tuned_model_1_epochs_300.model"
model2 = "fine_tuned_incremental_model_1_epochs_300.model"


create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_all.csv",align_embeddings=True)



######################################################################## Plot the annotation distribution ##################################################################################################################################################################################


import matplotlib.pyplot as plt

def analyze_annotation_distribution(df):
    """
    Analyze the distribution of annotations in the dataset.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing an 'annotation' column.

    Returns:
    None: Displays a distribution plot with percentages as labels and prints summary statistics.
    """
    # Count the occurrences of each annotation
    annotation_counts = df['annotation'].value_counts()
    annotation_percentages = (annotation_counts / len(df)) * 100

    # Combine counts and percentages into a single DataFrame for display
    annotation_summary = pd.DataFrame({
        'Count': annotation_counts,
        'Percentage': annotation_percentages
    })

    print("Annotation Distribution:")
    print(annotation_summary)

    # Plot the distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        annotation_summary.index,
        annotation_summary['Count'],
        color=['#007acc', '#ffa07a', '#90ee90'],  # Custom colors for better aesthetics
        edgecolor='black'
    )

    ax.set_title("Annotation Distribution", fontsize=16, fontweight='bold')
    ax.set_xlabel("Annotation", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_xticks(annotation_summary.index)
    ax.set_xticklabels(annotation_summary.index, fontsize=10)

    # Add percentages as labels on top of the bars
    for bar, count, pct in zip(bars, annotation_summary['Count'], annotation_summary['Percentage']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,  # Slightly above the bar
            f"{pct:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.show()

# Example usage (replace 'df' with your actual DataFrame):
analyze_annotation_distribution(filtered_df)








##########################################################################################################################################################################################################################################################

################################################## Create test set from ground truth ################################################################################################################################################################################

com = final_df["word"].tolist()

# Remove the common words from target_words
filtered_target_words = [word for word in target_words if word not in com]

# Display the filtered list
len(filtered_target_words)
len(target_words)



# Filter to keep only the common words
filtered_list = [item for item in concatenated_list if item[0] in filtered_target_words]

# Display the filtered list
len(filtered_list)




############## 1 original attemp 

# Convert to DataFrame
test_set_df = pd.DataFrame(concatenated_list, columns=['word', 'annotation'])

# Save DataFrame to CSV
test_set_df.to_csv("test_set.csv", index=False)


# Application 

annotated_csv = "test_set.csv"

model1 = "fine_tuned_model_1_epochs_300.model"
model2 = "fine_tuned_incremental_model_1_epochs_300.model"

create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_test_set.csv",align_embeddings=True)

#### A good idea would be to enhance the annotated dataset with synthetic data created manually like sentences with apple, gay etc known semantic change words 





df_test= pd.read_csv("final_features_test_set.csv")



##################2 trial with only 26 words 

# Convert to DataFrame
test_set_df = pd.DataFrame(filtered_list, columns=['word', 'annotation'])

# Save DataFrame to CSV
test_set_df.to_csv("test_set1.csv", index=False)

# Application 

annotated_csv = "test_set1.csv"

model1 = "fine_tuned_model_1_epochs_300.model"
model2 = "fine_tuned_incremental_model_1_epochs_300.model"

create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_test_set26.csv",align_embeddings=True)




##################3 trial with other model 1 

# Convert to DataFrame
test_set_df = pd.DataFrame(concatenated_list, columns=['word', 'annotation'])

# Save DataFrame to CSV
test_set_df.to_csv("test_set.csv", index=False)

# Application 

annotated_csv = "test_set.csv"

model1 = "fine_tuned_model_1_epochs_300.model"
model2 = "fine_tuned_incremental_model_1_epochs_300.model"

create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_test_set2.csv",align_embeddings=False)



################## 4 trial with other model 2 

# Convert to DataFrame
test_set_df = pd.DataFrame(concatenated_list, columns=['word', 'annotation'])

# Save DataFrame to CSV
test_set_df.to_csv("test_set.csv", index=False)

# Application 

annotated_csv = "test_set.csv"

model1 = "model_1_no_pretrained_epochs_300.model"
model2 = "model_2_no_pretrained_epochs_300.model"

create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_test_set4.csv",align_embeddings=True)




################## 5  trial with other model 3 

# Convert to DataFrame
test_set_df = pd.DataFrame(concatenated_list, columns=['word', 'annotation'])

# Save DataFrame to CSV
test_set_df.to_csv("test_set.csv", index=False)

# Application 

annotated_csv = "test_set.csv"

model1 = "model_1_no_pretrained_epochs_300.model"
model2 = "model_1_incremental_epochs_300.model"

create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_test_set5.csv",align_embeddings=True)




################## 6  trial with other model 4 ££££winner££££££ 

# Convert to DataFrame
test_set_df = pd.DataFrame(concatenated_list, columns=['word', 'annotation'])

# Save DataFrame to CSV
test_set_df.to_csv("test_set.csv", index=False)

# Application 

annotated_csv = "test_set.csv"

model1 = "model_1_no_pretrained_epochs_300.model"
model2 = "model_1_incremental_epochs_300.model"

create_features(annotated_csv, model1, model2, corpus_1_pos, corpus_2_pos, output_file="final_features_test_set6.csv",align_embeddings=False)



##########################################################################################################################################################################################################################################################
##############################################################Apply Supervised learning ############################################################################################################################################################################################




'''
Bagging:

Using BaggingClassifier with DecisionTreeClassifier as the base estimator.
Voting Classifier:

Hard Voting: Majority voting (votes from classifiers are hard, based on the predicted class).
Soft Voting: Classifier outputs probabilities, and the majority is taken based on probabilities.
Stacking Classifier:

Stack multiple classifiers like Random Forest, Gradient Boosting, and XGBoost, and use Logistic Regression as the final estimator to make the final predictions.
Notes:
For Voting Classifiers:

Hard Voting: The final prediction is based on majority voting from all classifiers.
Soft Voting: The final prediction is based on the average of predicted probabilities from each classifier.
For Stacking Classifier:

We stack classifiers like Random Forest, Gradient Boosting, and XGBoost, and use a Logistic Regression model as the final estimator to combine their predictions.

'''



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier



# Load the feature dataset for training and validation
df = pd.read_csv('final_features.csv')

# Separate features and target
X = df.drop(columns=['word', 'annotation'])  # Drop 'word' and keep 'annotation' as the target
y = df['annotation']

# Replace `inf` and `-inf` values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values in the features (Option 1: Imputation)
imputer = SimpleImputer(strategy='mean')  # You can change 'mean' to 'median' or another strategy
X = imputer.fit_transform(X)

# Split the data into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=42),
    "SVM": SVC(probability=True, max_iter=500, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "SGD Classifier": SGDClassifier(max_iter=1000, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    
    "Bagging (DecisionTree)": BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
    
    "Voting Classifier (Hard Voting)": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ], voting='hard'),
    
    "Voting Classifier (Soft Voting)": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ], voting='soft'),

    "Stacking Classifier": StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ], final_estimator=LogisticRegression(), cv=5)
}

# Create an empty list to store results for validation set
val_results = []

# Iterate over classifiers, train them, and store performance metrics on validation set
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_val_pred = clf.predict(X_val)
    y_val_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, 'predict_proba') else None  # Use predict_proba for ROC-AUC (if supported)
    
    # Calculate metrics for validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_val, y_val_pred, pos_label=1)
    f1 = f1_score(y_val, y_val_pred, pos_label=1)
    roc_auc = roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else "N/A"  # Some models don't support predict_proba
    
    # Append the results to the list
    val_results.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

# Convert the list of validation results into a DataFrame
val_results_df = pd.DataFrame(val_results)

# Display and save validation results
print(val_results_df)
val_results_df.to_csv('validation_results.csv', index=False)

# Select the best model based on F1-Score
best_model_name = val_results_df.loc[val_results_df['F1-Score'].idxmax()]['Classifier']
best_model = classifiers[best_model_name]

print(f"\nBest model based on F1-Score: {best_model_name}")

# Load the final unseen test set
test_set_df = pd.read_csv('final_features_test_set.csv')
X_test = test_set_df.drop(columns=['word', 'annotation'])
y_test = test_set_df['annotation']

# Handle any missing values in the test set and ensure feature names are retained
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)  # Ensure feature names are retained


# Evaluate all classifiers on the test set
test_results = []

for name, clf in classifiers.items():
    # Make predictions on the test set
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    # Calculate metrics for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
    test_roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else "N/A"

    # Append the results to the list
    test_results.append({
        "Classifier": name,
        "Accuracy": test_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1-Score": test_f1,
        "ROC-AUC": test_roc_auc
    })

# Convert the test results into a DataFrame
test_results_df = pd.DataFrame(test_results)

# Display and save test results
print("\nTest set results for all models:")
print(test_results_df)
test_results_df.to_csv('test_results_all_models.csv', index=False)



# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else "N/A"

# Save test set results
test_results = pd.DataFrame([{
    "Classifier": best_model_name,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1-Score": test_f1,
    "ROC-AUC": test_roc_auc
}])

print("\nTest set results for the best model:")
print(test_results)

# Save test set results to a CSV file
test_results.to_csv('test_results.csv', index=False)






######################################################################## Fine tune best model ############################################################################# 












from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
import pandas as pd
import joblib

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate the best model on the validation set
print(f"Evaluating the best model: {best_model_name}")
evaluate_model(best_model, X_train, X_val, y_train, y_val)

# Fine-Tuning with GridSearchCV for the Best Model
if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
elif best_model_name == "SVM":
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf']
    }
elif best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
elif best_model_name == "Decision Tree":
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }
else:
    param_grid = {}  # Empty grid if no parameters are defined

# Only perform GridSearchCV if there are parameters to tune
if param_grid:
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters from grid search
    print(f"Best parameters found by GridSearchCV for {best_model_name}:")
    print(grid_search.best_params_)
    
    # Update best_model with the fine-tuned version
    best_model = grid_search.best_estimator_

# Evaluate the fine-tuned model on the validation set
print(f"\nEvaluating the fine-tuned model: {best_model_name}")
evaluate_model(best_model, X_train, X_val, y_train, y_val)

# Perform cross-validation to ensure robustness
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
print(f"10-fold cross-validation accuracy for {best_model_name}: {np.mean(cv_scores):.3f}")

# Save the final feature matrix and the model for later use
pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]).to_csv('features_with_annotations.csv', index=False)

# Save the fine-tuned model to a file
joblib.dump(best_model, f'{best_model_name}_fine_tuned_model.pkl')
print(f"Fine-tuned model saved as {best_model_name}_fine_tuned_model.pkl")

# ================= Final Check on Test Set =================

# Load the unseen test set
test_set_df = pd.read_csv('test_set.csv')
X_test = test_set_df.drop(columns=['word', 'annotation'])
y_test = test_set_df['annotation']

# Handle any missing values in the test set
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)  # Ensure feature names are retained

# Evaluate the fine-tuned model on the unseen test set
print(f"\nFinal evaluation on the unseen test set with the fine-tuned {best_model_name}:")
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

# Calculate and print metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else "N/A"

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-Score:", test_f1)
print("Test ROC AUC:", test_roc_auc)

# Save test set results
test_results = pd.DataFrame([{
    "Classifier": best_model_name,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1-Score": test_f1,
    "ROC-AUC": test_roc_auc
}])

test_results.to_csv('final_test_results.csv', index=False)
print("\nTest set results saved as 'final_test_results.csv'")








