import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.covariance import empirical_covariance, MinCovDet



def kPCA(train_matrix, test_matrix):
    '''Compute kPCA for train and test'''
    k_PCA = KernelPCA(n_components=100, kernel='rbf')
    X_train = k_PCA.fit_transform(train_matrix)
    X_test = k_PCA.transform(test_matrix)
    return X_train, X_test


def separate_pos_neg(df_train, X_train):
    '''Create X_train_pos, X_train_neg'''
    labels = df_train['label']
    pos_labels = labels.reset_index(drop=True).astype(bool)
    neg_labels = pos_labels==False
    X_train_pos = np.delete(X_train, pos_labels, axis=0)
    X_train_neg = np.delete(X_train, neg_labels, axis=0)
    return X_train_pos, X_train_neg


def estimate_mean(X_train_label):
    '''Input : either X_train_pos or X_train_neg'''
    return np.mean(X_train_label, axis=0)

    
def estimate_cov(X_train_label):
    '''Input : either X_train_pos or X_train_neg'''
    return empirical_covariance(X_train_label)


def estimate_mcd(X_train_label):
    '''Input : either X_train_pos or X_train_neg'''
    mcd = MinCovDet().fit(X_train_label)
    return mcd.covariance_

    
def gaussian_likelihood(X_test, mean, cov):
  '''Input : 
  X_test : matrix of test embeddings
  mean : mean (for positive or negative distribution)
  cov : covariance (for positive or negative distribution)
  '''
  L = []
  for z in X_test:
    inv_cov = np.linalg.inv(cov)
    likelihood = - np.dot(np.dot(np.transpose(z - mean), inv_cov), z-mean)
    L.append(likelihood)
  return L

  
def compute_opposite_likelihood(X_test, pos_mean, neg_mean, pos_cov, neg_cov):
  '''Select max(positive likelihood, negative likelihood) and return (-1)*likelihood'''
  pos_likelihoods = gaussian_likelihood(X_test, pos_mean, pos_cov)
  neg_likelihoods = gaussian_likelihood(X_test, neg_mean, neg_cov)

  n = len(X_test)
  assert len(pos_likelihoods)==n
  assert len(neg_likelihoods)==n

  L = []
  for i in range(n):
    if pos_likelihoods[i] >= neg_likelihoods[i]:
      L.append(- pos_likelihoods[i]) # We want to algorithm to predict attacks when likelihood is the lower, i.e. (-1)*likelihood is the highest
    else:
      L.append(- neg_likelihoods[i])
  
  return L
  

def RDE_results(df_train, df_test, train_matrix, test_matrix):
    '''Input : 
    df_train: associate df_train in /pickle
    df_test: associate df_test in /pickle
    train_matrix: associate embeddings matrix in /embeddings
    test_matrix: associate embeddings matrix in /embeddings
    Output: results on test set
    '''
    X_train, X_test = kPCA(train_matrix, test_matrix)
    X_train_pos, X_train_neg = separate_pos_neg(df_train, X_train)
    
    pos_mean = estimate_mean(X_train_pos)
    pos_cov = estimate_cov(X_train_pos)
    pos_mcd = estimate_mcd(X_train_pos)
    
    neg_mean = estimate_mean(X_train_neg)
    neg_cov = estimate_cov(X_train_neg)
    neg_mcd = estimate_mcd(X_train_neg)
    
    OL_cov = compute_opposite_likelihood(X_test, pos_mean, neg_mean, pos_cov, neg_cov)
    OL_mcd = compute_opposite_likelihood(X_test, pos_mean, neg_mean, pos_mcd, neg_mcd)
    df_results = df_test.copy()
    df_results['OL_cov'] = OL_cov
    df_results['OL_mcd'] = OL_mcd
    df_results
    
