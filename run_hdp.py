import sys, os
from corpus import *
import hdp
import cPickle
import random, time
from numpy import cumsum, sum
from itertools import izip
from optparse import OptionParser
from glob import glob
np = hdp.np

def parse_args():
  parser = OptionParser()
  parser.set_defaults(T=300, K=20, D=-1, W=-1, eta=0.01, alpha=1.0, gamma=1.0,
                      max_time=100, max_iter=-1, var_converge=0.0001, random_seed=999931111, 
                      corpus_name=None, data_path=None, test_data_path=None, 
                      test_data_path_in_folds=None, directory=None)

  parser.add_option("--T", type="int", dest="T",
                    help="top level truncation [300]")
  parser.add_option("--K", type="int", dest="K",
                    help="second level truncation [20]")
  parser.add_option("--D", type="int", dest="D",
                    help="number of documents [-1]")
  parser.add_option("--W", type="int", dest="W",
                    help="size of vocabulary [-1]")
  parser.add_option("--eta", type="float", dest="eta",
                    help="the topic Dirichlet [0.01]")
  parser.add_option("--alpha", type="float", dest="alpha",
                    help="alpha value [1.0]")
  parser.add_option("--gamma", type="float", dest="gamma",
                    help="gamma value [1.0]")
  parser.add_option("--max_time", type="int", dest="max_time",
                    help="max time to run training in seconds [100]")
  parser.add_option("--max_iter", type="int", dest="max_iter",
                    help="max iteration to run training [-1]")
  parser.add_option("--var_converge", type="float", dest="var_converge",
                    help="relative change on doc lower bound [0.0001]")
  parser.add_option("--random_seed", type="int", dest="random_seed",
                    help="the random seed [999931111]")
  parser.add_option("--corpus_name", type="string", dest="corpus_name",
                    help="the corpus name: nature, nyt or wiki [None]")
  parser.add_option("--data_path", type="string", dest="data_path",
                    help="training data path or pattern [None]")
  parser.add_option("--test_data_path", type="string", dest="test_data_path",
                    help="testing data path [None]")
  parser.add_option("--test_data_path_in_folds", type="string",
                    dest="test_data_path_in_folds",
                    help="testing data prefix for different folds [None]")
  parser.add_option("--directory", type="string", dest="directory",
                    help="output directory [None]")

  (options, args) = parser.parse_args()
  return options 

def run_hdp():
  options = parse_args()
  random.seed(options.random_seed)
  # Read the training data.
  c_train_filename = options.data_path

  test_data_path = options.test_data_path
  c_test = read_data(test_data_path)
  c_test_word_count = sum([doc.total for doc in c_test.docs])

  if options.test_data_path_in_folds is not None:
    test_data_path_in_folds = options.test_data_path_in_folds
    test_data_in_folds_filenames = glob(test_data_path_in_folds)
    test_data_in_folds_filenames.sort()
    num_folds = len(test_data_in_folds_filenames)/2
    test_data_train_filenames = []
    test_data_test_filenames = []

    for i in range(num_folds):
      test_data_train_filenames.append(test_data_in_folds_filenames[2*i+1])
      test_data_test_filenames.append(test_data_in_folds_filenames[2*i])

    c_test_train_folds = [read_data(filename) for filename in test_data_train_filenames]
    c_test_test_folds = [read_data(filename) for filename in test_data_test_filenames]

  result_directory = "%s/corpus-%s" % (options.directory, options.corpus_name)
  print "creating directory %s" % result_directory
  if not os.path.isdir(result_directory):
    os.makedirs(result_directory)

  options_file = file("%s/options.dat" % result_directory, "w")
  for opt, value in options.__dict__.items():
    options_file.write(str(opt) + " " + str(value) + "\n")
  options_file.close()

  print "creating hdp instance."

  bhdp_hp = hdp.hdp_hyperparameter(options.alpha, options.alpha, options.gamma, options.gamma, False)

  bhdp = hdp.hdp(options.T, options.K, options.D, options.W, options.eta, bhdp_hp)
  #bhdp.seed_init(c_train)

  print "setting up counters and log files."
  iter = 0 
  total_time = 0.0
  total_doc_count = 0

  likelihood = 0.0
  old_likelihood = 0.0
  converge = 1.0

  log_file = file("%s/log.dat" % result_directory, "w") 
  log_file.write("iteration time doc.count likelihood\n")

  test_log_file = file("%s/test-log.dat" % result_directory, "w") 
  test_log_file.write("iteration time doc.count score word.count score.split word.count.split\n")

  while (options.max_iter == -1 or iter < options.max_iter) and total_time < options.max_time:
    t0 = time.clock()
    # Run one step iteration.
    likelihood = bhdp.em_on_large_data(c_train_filename, options.var_converge, fresh=(iter==0))
    if iter > 0:
        converge = (likelihood - old_likelihood)/abs(old_likelihood)
    old_likelihood = likelihood
    print "iter = %d, likelihood = %f, converge = %f" % (iter, likelihood, converge)
    if converge < 0:
        print "warning, likelihood is decreasing!"

    total_time += time.clock() - t0

    iter += 1  # increase the iter counter
    total_doc_count += options.D

    log_file.write("%d %d %d %.5f\n" % (iter, total_time, total_doc_count, likelihood))
    log_file.flush()

    bhdp.save_topics('%s/doc_count-%d.topics' %  (result_directory, total_doc_count))
    cPickle.dump(bhdp, file('%s/doc_count-%d.model' % (result_directory, total_doc_count), 'w'), -1)

    print "\tworking on predictions."
    (lda_alpha, lda_beta) = bhdp.hdp_to_lda()

    # prediction on the fixed test in folds
    print "\tworking on fixed test data."
    test_score = 0.0
    test_score_split = 0.0
    c_test_word_count_split = 0
    for doc in c_test.docs:
      (likelihood, gamma) = hdp.lda_e_step(doc, lda_alpha, lda_beta)
      test_score += likelihood
      (likelihood, count, gamma) = hdp.lda_e_step_split(doc, lda_alpha, lda_beta)
      test_score_split += likelihood
      c_test_word_count_split += count

    test_log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
                        total_doc_count, test_score, c_test_word_count, 
                        test_score_split, c_test_word_count_split))
    test_log_file.flush()

    # prediction on the test set in the folds
   # print "\tworking on test data in folds."
   # test_folds_log_file = file("%s/doc_count-%d.test.folds" % (result_directory, total_doc_count), "w")
   # test_folds_log_file.write("fold doc.id word count score\n")
   # for i in range(num_folds):
   #   train_data = c_test_train_folds[i]
   #   test_data = c_test_test_folds[i]
   #   for (doc_id, train_doc, test_doc) in izip(range(train_data.num_docs), train_data.docs, test_data.docs):
   #     if test_doc.total > 0:
   #       (likelihood, gamma) = hdp.lda_e_step(train_doc, lda_alpha, lda_beta)
   #       theta = gamma/np.sum(gamma)
   #       lda_betad = lda_beta[:, test_doc.words]
   #       log_predicts = np.log(np.dot(theta, lda_betad))
   #       log_info = "\n".join(["%d %d %d %d %.5f" % (i, doc_id, word, word_count, f) for (word, word_count, f) in izip(test_doc.words, test_doc.counts, log_predicts)])
   #       test_folds_log_file.write(log_info + "\n") 

   # test_folds_log_file.close()

  log_file.close()

  print "Saving the final model and topics."
  bhdp.save_topics('%s/final.topics' %  result_directory)
  cPickle.dump(bhdp, file('%s/final.model' % result_directory, 'w'), -1)

  (lda_alpha, lda_beta) = bhdp.hdp_to_lda()

  # prediction on the fixed test in folds
  print "\tworking on fixed test data."
  test_score = 0.0
  test_score_split = 0.0
  c_test_word_count_split = 0
  for doc in c_test.docs:
    (likelihood, gamma) = hdp.lda_e_step(doc, lda_alpha, lda_beta)
    test_score += likelihood
    (likelihood, count, gamma) = hdp.lda_e_step_split(doc, lda_alpha, lda_beta)
    test_score_split += likelihood
    c_test_word_count_split += count

  test_log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
                      total_doc_count, test_score, c_test_word_count, 
                      test_score_split, c_test_word_count_split))
  test_log_file.flush()

  test_log_file.close()

if __name__ == '__main__':
  run_hdp()
