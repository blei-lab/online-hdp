# the hdp class in python
# implements the truncated hdp model
# by chongw@cs.princeton.edu

import numpy as np
import scipy.special as sp
import os, sys, math, time
import utils
from corpus import document, corpus, parse_line
from itertools import izip
import random

meanchangethresh = 0.00001
random_seed = 999931111
np.random.seed(random_seed)
random.seed(random_seed)

def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
    return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])

def expect_log_sticks(sticks):
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks 

def lda_e_step_half(doc, alpha, Elogbeta, split_ratio):

    n_train = int(doc.length * split_ratio)
    n_test = doc.length - n_train
   
   # split the document
    words_train = doc.words[:n_train]
    counts_train = doc.counts[:n_train]
    words_test = doc.words[n_train:]
    counts_test = doc.counts[n_train:]
    
    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 

    expElogbeta = np.exp(Elogbeta[:, words_train])
    phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    max_iter = 100
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  expElogbeta.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break
    gamma = gamma/np.sum(gamma)
    counts = np.array(counts_test)
    expElogbeta = np.exp(Elogbeta[:, words_test])
    score = np.sum(counts * np.log(np.dot(gamma, expElogbeta) + 1e-100))

    return (score, np.sum(counts), gamma)

def lda_e_step_split(doc, alpha, beta, max_iter=100):
    half_len = int(doc.length/2) + 1
    idx_train = [2*i for i in range(half_len) if 2*i < doc.length]
    idx_test = [2*i+1 for i in range(half_len) if 2*i+1 < doc.length]
   
   # split the document
    words_train = [doc.words[i] for i in idx_train]
    counts_train = [doc.counts[i] for i in idx_train]
    words_test = [doc.words[i] for i in idx_test]
    counts_test = [doc.counts[i] for i in idx_test]

    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break

    gamma = gamma/np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)

def lda_e_step(doc, alpha, Elogbeta, max_iter=100):
    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    expElogbeta = Elogbeta[:, doc.words]
    phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
    counts = np.array(doc.counts)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  expElogbeta.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha-gamma) * Elogtheta)
    likelihood += np.sum(sp.gammaln(gamma) - sp.gammaln(alpha))
    likelihood += sp.gammaln(np.sum(alpha)) - sp.gammaln(np.sum(gamma))

    return (likelihood, gamma)

class hdp_hyperparameter:
    def __init__(self, alpha_a, alpha_b, gamma_a, gamma_b, hyper_opt=False):
        self.m_alpha_a = alpha_a
        self.m_alpha_b = alpha_b
        self.m_gamma_a = gamma_a
        self.m_gamma_b = gamma_b
        self.m_hyper_opt = hyper_opt 

class suff_stats:
    def __init__(self, T, size_vocab):
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_beta_ss = np.zeros((T, size_vocab))
    
    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)

class hdp:
    ''' hdp model using john's new stick breaking'''
    def __init__(self, T, K, D,  size_vocab, eta, hdp_hyperparam):
        ''' this follows the convention of the HDP paper'''
        ''' gamma, first level concentration ''' 
        ''' alpha, second level concentration '''
        ''' eta, the topic Dirichlet '''
        ''' T, top level truncation level '''
        ''' K, second level truncation level '''
        ''' size_vocab, size of vocab'''
        ''' hdp_hyperparam, the hyperparameter of hdp '''
    
        self.m_hdp_hyperparam = hdp_hyperparam

        self.m_T = T
        self.m_K = K # for now, we assume all the same for the second level truncation
        self.m_size_vocab = size_vocab

        self.m_beta = np.random.gamma(1.0, 1.0, (T, size_vocab)) * D*100/(T*size_vocab)
        self.m_eta = eta

        self.m_alpha = hdp_hyperparam.m_alpha_a/hdp_hyperparam.m_alpha_b
        self.m_gamma = hdp_hyperparam.m_gamma_a/hdp_hyperparam.m_gamma_b
        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        self.m_var_sticks[1] = self.m_gamma

        # variational posterior parameters for hdp
        self.m_var_gamma_a = hdp_hyperparam.m_gamma_a
        self.m_var_gamma_b = hdp_hyperparam.m_gamma_b
   
    def save_topics(self, filename):
        f = file(filename, "w") 
        for beta in self.m_beta:
            line = ' '.join([str(x) for x in beta])  
            f.write(line + '\n')
        f.close()

    def doc_e_step(self, doc, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=False):

        Elogbeta_doc = Elogbeta[:, doc.words] 
        v = np.zeros((2, self.m_K-1))

        phi = np.ones((doc.length, self.m_K)) * 1.0/self.m_K

        # the following line is of no use
        Elogsticks_2nd = expect_log_sticks(v)

        likelihood = 0.0
        old_likelihood = -1e1000
        converge = 1.0 
        eps = 1e-100
        
        iter = 0
        max_iter = 100
        #(TODO): support second level optimization in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi 
            if iter < 3 and fresh:
                var_phi = np.dot(phi.T, (Elogbeta_doc * doc.counts).T)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T, (Elogbeta_doc * doc.counts).T) + Elogsticks_1st
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)

           # phi
            if iter < 3:
                phi = np.dot(var_phi, Elogbeta_doc).T
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)
            else: 
                phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc.counts))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < 0:
                print "warning, likelihood is decreasing!"
            
            iter += 1
            
        # update the suff_stat ss 
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_beta_ss[:, doc.words] += np.dot(var_phi.T, phi.T * doc.counts)

        return(likelihood)

    def optimal_ordering(self, ss):
        s = [(a, b) for (a,b) in izip(ss.m_var_sticks_ss, range(self.m_T))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        ss.m_var_sticks_ss[:] = ss.m_var_sticks_ss[idx]
        ss.m_var_beta_ss[:] = ss.m_var_beta_ss[idx,:]

    def do_m_step(self, ss):
        self.optimal_ordering(ss)
        ## update top level sticks 
        self.m_var_sticks[0] = ss.m_var_sticks_ss[:self.m_T-1] + 1.0
        var_phi_sum = np.flipud(ss.m_var_sticks_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma 

        ## update topic parameters
        self.m_beta = self.m_eta + ss.m_var_beta_ss
        
        if self.m_hdp_hyperparam.m_hyper_opt:
            self.m_var_gamma_a = self.m_hdp_hyperparam.m_gamma_a + self.m_T - 1
            dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
            Elog1_W = sp.psi(self.m_var_sticks[1]) - dig_sum
            self.m_var_gamma_b = self.m_hdp_hyperparam.m_gamma_b - np.sum(Elog1_W)
            self.m_gamma = hdp_hyperparam.m_gamma_a/hdp_hyperparam.m_gamma_b

    def seed_init(self, c):
        n = c.num_docs
        ids = random.sample(range(n), self.m_T) 
        print "seeding with docs %s" % (' '.join([str(id) for id in ids]))
        for (id, t) in izip(ids, range(self.m_T)):
            doc = c.docs[id]
            self.m_beta[t] = np.random.gamma(1, 1, self.m_size_vocab) 
            self.m_beta[t,doc.words] += doc.counts 

    ## one iteration of the em
    def em_on_large_data(self, filename, var_converge, fresh):
        ss = suff_stats(self.m_T, self.m_size_vocab)
        ss.set_zero()
        
        # prepare all needs for a single doc
        Elogbeta = dirichlet_expectation(self.m_beta) # the topics
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        likelihood = 0.0
        for line in file(filename):
            doc = parse_line(line)
            likelihood += self.doc_e_step(doc, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=fresh)
        
        # collect the likelihood from other parts
        # the prior for gamma
        if self.m_hdp_hyperparam.m_hyper_opt:
            log_gamma = sp.psi(self.m_var_gamma_a) -  np.log(self.m_var_gamma_b)
            likelihood += self.m_hdp_hyperparam.m_gamma_a * log(self.m_hdp_hyperparam.m_gamma_b) \
                    - sp.gammaln(self.m_hdp_hyperparam.m_gamma_a) 

            likelihood -= self.m_var_gamma_a * log(self.m_var_gamma_b) \
                    - sp.gammaln(self.m_var_gamma_a) 

            likelihood += (self.m_hdp_hyperparam.m_gamma_a - self.m_var_gamma_a) * log_gamma \
                    - (self.m_hdp_hyperparam.m_gamma_b - self.m_var_gamma_b) * self.m_gamma
        else:
            log_gamma = np.log(self.m_gamma)

       # the W/sticks part 
        likelihood += (self.m_T-1) * log_gamma
        dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
        likelihood += np.sum((np.array([1.0, self.m_gamma])[:,np.newaxis] - self.m_var_sticks) * (sp.psi(self.m_var_sticks) - dig_sum))
        likelihood -= np.sum(sp.gammaln(np.sum(self.m_var_sticks, 0))) - np.sum(sp.gammaln(self.m_var_sticks))
        
        # the beta part    
        likelihood += np.sum((self.m_eta - self.m_beta) * Elogbeta)
        likelihood += np.sum(sp.gammaln(self.m_beta) - sp.gammaln(self.m_eta))
        likelihood += np.sum(sp.gammaln(self.m_eta*self.m_size_vocab) - sp.gammaln(np.sum(self.m_beta, 1)))

        self.do_m_step(ss) # run m step
        return likelihood
    
    ## one iteration of the em
    def em(self, c, var_converge, fresh):
        ss = suff_stats(self.m_T, self.m_size_vocab)
        ss.set_zero()
        
        # prepare all needs for a single doc
        Elogbeta = dirichlet_expectation(self.m_beta) # the topics
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        likelihood = 0.0
        for doc in c.docs:
            likelihood += self.doc_e_step(doc, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=fresh)
        
        # collect the likelihood from other parts
        # the prior for gamma
        if self.m_hdp_hyperparam.m_hyper_opt:
            log_gamma = sp.psi(self.m_var_gamma_a) -  np.log(self.m_var_gamma_b)
            likelihood += self.m_hdp_hyperparam.m_gamma_a * log(self.m_hdp_hyperparam.m_gamma_b) \
                    - sp.gammaln(self.m_hdp_hyperparam.m_gamma_a) 

            likelihood -= self.m_var_gamma_a * log(self.m_var_gamma_b) \
                    - sp.gammaln(self.m_var_gamma_a) 

            likelihood += (self.m_hdp_hyperparam.m_gamma_a - self.m_var_gamma_a) * log_gamma \
                    - (self.m_hdp_hyperparam.m_gamma_b - self.m_var_gamma_b) * self.m_gamma
        else:
            log_gamma = np.log(self.m_gamma)

       # the W/sticks part 
        likelihood += (self.m_T-1) * log_gamma
        dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
        likelihood += np.sum((np.array([1.0, self.m_gamma])[:,np.newaxis] - self.m_var_sticks) * (sp.psi(self.m_var_sticks) - dig_sum))
        likelihood -= np.sum(sp.gammaln(np.sum(self.m_var_sticks, 0))) - np.sum(sp.gammaln(self.m_var_sticks))
        
        # the beta part    
        likelihood += np.sum((self.m_eta - self.m_beta) * Elogbeta)
        likelihood += np.sum(sp.gammaln(self.m_beta) - sp.gammaln(self.m_eta))
        likelihood += np.sum(sp.gammaln(self.m_eta*self.m_size_vocab) - sp.gammaln(np.sum(self.m_beta, 1)))

        self.do_m_step(ss) # run m step
        return likelihood
                
    def hdp_to_lda(self):
        # compute the lda almost equivalent hdp.
        # alpha
        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        alpha = alpha * self.m_alpha
        #alpha = alpha * self.m_gamma
        
        # beta
        beta_sum = np.sum(self.m_beta, axis=1)
        beta = self.m_beta / beta_sum[:, np.newaxis]

        return (alpha, beta)
         
    def em_with_testing(self, c, max_iter, var_converge, max_time, directory, c_test, split_ratio, seeded=True):
        ## the em style inference 
        if seeded:
            self.seed_init(c) 

        ss = suff_stats(self.m_T, self.m_size_vocab)
        
        likelihood = 0.0
        old_likelihood = 0.0
        converge = 1.0

        out_predict = file('%s/hdp.predict' % directory, "w")

        iter = 0 
        totaltime = 0.0
        while (max_iter == -1 or iter < max_iter) and totaltime < max_time:

            t0 = time.clock()
            # prepare all needs for a single doc
            Elogbeta = dirichlet_expectation(self.m_beta) # the topics
            Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
            ss.set_zero()
            likelihood = 0.0
            for doc in c.docs:
                likelihood += self.doc_e_step(doc, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=(iter==0))
            
            # collect the likelihood from other parts
            # the prior for gamma
            if self.m_hdp_hyperparam.m_hyper_opt:
                log_gamma = sp.psi(self.m_var_gamma_a) -  np.log(self.m_var_gamma_b)
                likelihood += self.m_hdp_hyperparam.m_gamma_a * log(self.m_hdp_hyperparam.m_gamma_b) \
                        - sp.gammaln(self.m_hdp_hyperparam.m_gamma_a) 

                likelihood -= self.m_var_gamma_a * log(self.m_var_gamma_b) \
                        - sp.gammaln(self.m_var_gamma_a) 

                likelihood += (self.m_hdp_hyperparam.m_gamma_a - self.m_var_gamma_a) * log_gamma \
                        - (self.m_hdp_hyperparam.m_gamma_b - self.m_var_gamma_b) * self.m_gamma
            else:
                log_gamma = np.log(self.m_gamma)

           # the W/sticks part 
            likelihood += (self.m_T-1) * log_gamma
            dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
            likelihood += np.sum((np.array([1.0, self.m_gamma])[:,np.newaxis] - self.m_var_sticks) * (sp.psi(self.m_var_sticks) - dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(self.m_var_sticks, 0))) - np.sum(sp.gammaln(self.m_var_sticks))
            
            # the beta part    
            likelihood += np.sum((self.m_eta - self.m_beta) * Elogbeta)
            likelihood += np.sum(sp.gammaln(self.m_beta) - sp.gammaln(self.m_eta))
            likelihood += np.sum(sp.gammaln(self.m_eta*self.m_size_vocab) - sp.gammaln(np.sum(self.m_beta, 1)))
            
            if iter > 0:
                converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            print "iter = %d, likelihood = %f, converge = %f" % (iter, likelihood, converge)

            if converge < 0:
                print "warning, likelihood is decreasing!"

            self.do_m_step(ss) # run m step
            iter += 1  # increase the iter counter
                
            totaltime += time.clock() - t0
            (score, nwords) = self.infer_only(c_test.docs, half_train_half_test=True, split_ratio=split_ratio)
            out_predict.write("%f %f\n" % (totaltime, score/nwords))
            out_predict.flush()
        out_predict.close()
              
    def infer_only(self, docs, half_train_half_test=False, split_ratio=0.5):
        #Elogbeta = dirichlet_expectation(self.m_beta) # the topics
        Elogbeta = np.log(self.m_beta) - np.log(np.sum(self.m_beta, 1))[:, np.newaxis]
        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        alpha = alpha * self.m_alpha
        #alpha = alpha * self.m_gamma
        score = 0.0
        count = 0.0
        for doc in docs:
            if half_train_half_test:
                (s, c, gamma) = lda_e_step_half(doc, alpha, Elogbeta, split_ratio) 
                score += s
                count += c
            else:
                (s, gamma) = lda_e_step(doc, alpha, Elogbeta)
                score += s 
                count += doc.total
        return (score, count)

