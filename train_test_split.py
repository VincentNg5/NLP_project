import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import accuracy_score

def create_filename(transformers, attack):
    '''
    Input: 
    transformers = {'bert', 'roberta'}
    attack = {'textfooler', 'pwws', 'bae', 'tf-adj'}
    Output : directory of the test set .csv
    '''
    full_transformers_name = 'bert-base-uncased' if transformers=='bert' else 'roberta-base'
    filename = 'data/imdb/' + transformers + '/' + attack + '/' + full_transformers_name + '-imdb_' + attack + '.csv'
    return filename
    
def generate_test_set(transformers, attack, n_max_adv=2000, seed=1803):
    # Create DataFrame
    filename = create_filename(transformers, attack)
    print('Loading from', filename)
    df = pd.read_csv(filename, sep=',')
    
    def clean_text(text):
        text = text.replace("[[", "")
        text = text.replace("]]", "")
        return text
        
    df['original_text'] = df['original_text'].map(clean_text)
    df['perturbed_text'] = df['perturbed_text'].map(clean_text)
    n_test = len(df)
    
    # Generate Scenario 1
    np.random.seed(seed)
    indices = np.random.permutation(n_test)
    indices_adv = indices[:n_max_adv]
    indices_clean = indices[n_max_adv:]
    
    df_adv = df.loc[indices_adv,:]
    mask_1 = df_adv['result_type']=='Successful' # Select only successful attacks
    mask_2 = df_adv['original_output'] == df_adv['ground_truth_output'] # Select only correctly classified examples
    df_adv = df_adv[mask_1 & mask_2]
    n_adv = len(df_adv) # Number of attack samples

    indices_clean_samples = np.random.choice(indices_clean, n_adv)
    df_clean = df.loc[indices_clean_samples,:]
    
    # Creating final DataFrame
    adv_text = df_adv['perturbed_text']
    adv_bool = np.ones(len(adv_text), dtype=int)
    clean_text = df_clean['original_text']
    clean_bool = np.zeros(len(clean_text), dtype=int)

    text_test = pd.concat([adv_text, clean_text])
    bool_test = np.concatenate([adv_bool, clean_bool])
    df_test = pd.DataFrame({'text': text_test, 'adversarial': bool_test})
    
    pickle_filename = 'pickle/imdb_' + transformers + '_test_' + attack + '.pickle'
    df_test.to_pickle(pickle_filename)
    print(pickle_filename, 'successfully created')
    
    return None
    