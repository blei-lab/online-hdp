#! /usr/bin/python

''' several useful functions '''
import numpy as np

def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def log_sum(log_a, log_b):
	''' we know log(a) and log(b), compute log(a+b) '''
	v = 0.0;
	if (log_a < log_b):
		v = log_b+np.log(1 + np.exp(log_a-log_b))
	else:
		v = log_a+np.log(1 + np.exp(log_b-log_a))
	return v


def argmax(x):
	''' find the index of maximum value '''
	n = len(x)
	val_max = x[0]
	idx_max = 0

	for i in range(1, n):
		if x[i]>val_max:
			val_max = x[i]
			idx_max = i		

	return idx_max			
