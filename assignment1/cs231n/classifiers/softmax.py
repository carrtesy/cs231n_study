import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    unnorm_log_prob = X[i].dot(W)
    unnorm_log_prob = unnorm_log_prob - np.max(unnorm_log_prob)
    unnorm_prob = np.exp(unnorm_log_prob)
    total = np.sum(unnorm_prob)
    norm_prob = unnorm_prob / total
    loss += -np.log(norm_prob[y[i]])
    for j in xrange(num_classes):
        dW[:, j] += unnorm_prob[j] * X[i] / total
    dW[:, y[i]] -= X[i]
  
  loss /= num_train
  loss += reg * 0.5 * np.sum(W*W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  unnorm_log_prob = X.dot(W)
  unnorm_log_prob = unnorm_log_prob - np.max(unnorm_log_prob, axis = -1, keepdims = True)
  unnorm_prob = np.exp(unnorm_log_prob)
  total = np.sum(unnorm_prob, axis = 1).reshape(num_train, 1)

  norm_prob = unnorm_prob / total
  loss += np.mean(-np.log(norm_prob[np.arange(num_train), y]))
  loss += reg * 0.5 * np.sum(W*W)
  norm_prob[np.arange(num_train), y] -= 1
  dW = (X.T).dot(norm_prob)
  dW /= num_train
  dW += reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

