import tensorflow as tf
import numpy as np

def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(x-y)
    d = tf.sqrt(tf.reduce_sum(d, axis=1)+1.0e-8) # What about the axis ???
    return d


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):

    """
    Compute the contrastive loss as in
    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m
    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin
    **Returns**
     Return the loss operation
    """

    with tf.name_scope("triplet_loss"):
        d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
        d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        #loss = d_p_squared - d_n_squared + margin

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)
    


def triplet_softmax_cross_entropy(preds, labels, triplet, mask, MARGIN, triplet_lamda):
    """Softmax cross-entropy loss with masking."""
    #origin
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#    train_anchor = preds[triplet[:,0],:]
 #   train_positive = preds[triplet[:,1],:]
  #  train_negative = preds[triplet[:,2],:]
    anchor, positive, negative = tf.unstack(triplet, 3, axis = 1)
    train_anchor = tf.gather(preds, anchor)
    train_positive = tf.gather(preds, positive)
    train_negative = tf.gather(preds, negative)
    loss_triple, positives, negatives = compute_triplet_loss(train_anchor, train_positive, train_negative, MARGIN)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = loss + triplet_lamda*loss_triple
    loss *= mask
    
    return tf.reduce_mean(loss)

def triplet_normalize_softmax_cross_entropy(preds, labels, triplet, mask, MARGIN, triplet_lamda):
    """Softmax cross-entropy loss with masking."""
    #origin
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#    train_anchor = preds[triplet[:,0],:]
 #   train_positive = preds[triplet[:,1],:]
  #  train_negative = preds[triplet[:,2],:]
    anchor, positive, negative = tf.unstack(triplet, 3, axis = 1)
    preds = tf.nn.l2_normalize(preds, 1)
    train_anchor = tf.gather(preds, anchor)
    train_positive = tf.gather(preds, positive)
    train_negative = tf.gather(preds, negative)
    loss_triple, positives, negatives = compute_triplet_loss(train_anchor, train_positive, train_negative, MARGIN)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = loss + triplet_lamda*loss_triple
    loss *= mask
    
    return tf.reduce_mean(loss)

def triplet_no_w1_softmax_cross_entropy(preds, return_no_w1, labels, triplet, mask, MARGIN, triplet_lamda):
    """Softmax cross-entropy loss with masking."""
    #origin
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#    train_anchor = preds[triplet[:,0],:]
 #   train_positive = preds[triplet[:,1],:]
  #  train_negative = preds[triplet[:,2],:]
    anchor, positive, negative = tf.unstack(triplet, 3, axis = 1)
    train_anchor = tf.gather(return_no_w1, anchor)
    train_positive = tf.gather(return_no_w1, positive)
    train_negative = tf.gather(return_no_w1, negative)
    loss_triple, positives, negatives = compute_triplet_loss(train_anchor, train_positive, train_negative, MARGIN)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = loss + triplet_lamda*loss_triple
    loss *= mask
    
    return tf.reduce_mean(loss)

def triplet_no_w1_normalize_softmax_cross_entropy(preds, return_no_w1, labels, triplet, mask, MARGIN, triplet_lamda):
    """Softmax cross-entropy loss with masking."""
    #origin
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#    train_anchor = preds[triplet[:,0],:]
 #   train_positive = preds[triplet[:,1],:]
  #  train_negative = preds[triplet[:,2],:]
    anchor, positive, negative = tf.unstack(triplet, 3, axis = 1)
    return_no_w1 = tf.nn.l2_normalize(return_no_w1, 1)
    train_anchor = tf.gather(return_no_w1, anchor)
    train_positive = tf.gather(return_no_w1, positive)
    train_negative = tf.gather(return_no_w1, negative)
    loss_triple, positives, negatives = compute_triplet_loss(train_anchor, train_positive, train_negative, MARGIN)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = loss + triplet_lamda*loss_triple
    loss *= mask
    
    return tf.reduce_mean(loss)

def weighted_softmax_cross_entropy(preds, labels, beta):
    print('A')
    # S = mask.reduce_sum(axis=0)
    # Sm = S.max()
    # S = (Sm-S)/beta*Sm + 1
    # loss *= (labels*S).reduce_sum(axis=1)
    y=tf.nn.softmax(preds)
    y_=tf.cast(labels, dtype=tf.float32)
    cross_entropy = y_ * tf.log(y)
#    print('cross_entropy', cross_entropy.shape)

    #build weight matrix
    sum = tf.reduce_sum(labels, axis=0)   #sum by row
    smax = tf.reduce_max(sum)
    smax_tensor=tf.fill([1, 7], smax)
    weighted = (smax_tensor-sum)/(beta*smax_tensor) + 1
    weighted = tf.cast(weighted, dtype=tf.float32)

    #calculate weighted-cross-entropy
    w_cross_entropy = weighted * cross_entropy
#    print('w_cross_entropy',w_cross_entropy.shape)
    loss=-1*tf.reduce_sum(w_cross_entropy, axis=1)
    print(loss)
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    #origin
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels) 
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
