"""This is a simple way to fix typos based on edit_distance similarity. It's a very 
simple and brute force algorithm: If two words are within max_distance from each 
other, the word with fewer occurrences will be mapped to the word with 
higher number of occurrences.


Key points:
- This may be useful when you have some string data in a column that has some typos.
- I would want to change edit_distance metric in many situations*


* edit_distance doesn't understand common words. Maybe using some tfidf or deep learning 
would be better for a similarity metric. In the dataset I first used

Takes ~3 minutes for 2 iterations on 1300 -> 1000 classes for my computer (hp i7 16gb ram). 
"""
from functools import partial


def edit_distance(word1, word2):
    """ Calculate edit distance between two words
    word1  (str): First word.
    word2  (str): Second word.
    """
    if word1 == '':
        return len(word2)
    if word2 == '':
        return len(word1)

    m, n = len(word1) + 1, len(word2) + 1

    memo = [[0] * n for _ in range(m)]

    # Initialize first row and first column. These represent the distances from empty string
    memo[0] = list(range(n))
    for i in range(m):
        memo[i][0] = i

    # memo[i][j] distance from word1[:i] and word2[:j]
    for i in range(1, m):
        for j in range(1, n):
            # Replace or keep:
            replace = memo[i - 1][j - 1] + int(word1[i - 1] != word2[j - 1])

            # Delete:
            delete = memo[i][j - 1] + 1

            #Insert:
            insert = memo[i - 1][j] + 1

            memo[i][j] = min((replace, delete, insert))

    return memo[m - 1][n - 1]


def fix_typo(df, target, min_len=5, max_distance=2, n_iters=2):
    """ Fix typo based on edit_distance similarity. If two words are
    within max_distance from each other, the word with fewer occurrences will 
    be mapped to the word with higher number of occurrences. 
    
    df     (pandas.DataFrame): DataFrame containing the column to fix typos in.

    target              (str): The name of the column containing the string data.
    
    min_len             (int): Minimum length of string to consider.
    
    max_distance        (int): Maximum edit_distance to consider as the same word.
    
    n_iters             (int): Number of iterations to repeat algorithm.
    """
    
    # Only rows with minimum length strings
    filtered_df = df[df[target].str.len() > min_len]
    
    # Iterate multiple times because we can go from wotlf bank -> wotld bank -> world bank 
    for iter_num in range(n_iters):
        # Mapping of incorrect words -> correct words
        correction_dict = {}

        # Count occurrences for each word
        corpus_weights = filtered_df.groupby(target)[target].count()
        groups_before = corpus_weights.shape[0] # Keep track of number of groups
        
        # Unique strings in our column
        corpus = pd.DataFrame(corpus_weights.index)

        # Iterate through corpus
        for i, word in enumerate(corpus[target]):
            # Calculate distance between each word in the corpus
            corpus['distance'] = corpus.loc[i:, target].apply(partial(edit_distance, word))
            corpus['weight'] = corpus[target].map(corpus_weights)
            
            # Apply the max distance threshold
            mispelled_df = corpus[corpus['distance'] <= max_distance]
            
            # The correct word will be the one with the most occurrences
            correct_word = mispelled_df.loc[mispelled_df['weight'].idxmax(), target]
            
            # Map the mispelled words to the correct word
            for mispelled_word in mispelled_df[target]:
                correction_dict[mispelled_word] = correct_word

        # Apply correction map
        filtered_df[target] = filtered_df[target].map(correction_dict)
        
        groups_after = filtered_df.groupby(target)[target].count().shape[0]
        print(f'Groups: {groups_before} --> {groups_after}')
            
    # Fill in the new mapping and return
    df.loc[filtered_df.index, target] = filtered_df[target]
    return df


# Example of manual finishing touches
target = 'installer'
min_count = 20

target_counts = df[target].value_counts() 

for val in target_counts[target_counts > min_count].index.sort_values():
    print(val)

manual_corrections = [
    ('adra', 'adra'),
    ('central gov', 'central government'),
    ('commu', 'community'),
    ('danid', 'danida'),
    ('distri', 'district council'),
    ('district water', 'district water department'),
    ('gov', 'government'),
    ('halmashauri', 'halmashauri'),
    ('isf', 'isf'),
    ('kkt', 'kkkt'),
    ('kkkt', 'kkkt'),
    ('missi', 'mission'),
    ('mwe', 'mwe'),
    ('nora', 'norad'),
    ('rc', 'rc church'),
    ('rwe', 'rwe'),
    ('nora', 'norad'),
    ('villa', 'villagers'),
    ('water', 'water aid'),
]

# Requires no NANs
for correct_from, correct_to in manual_corrections:
    df.loc[df[target].str.startswith(correct_from), target] = correct_to