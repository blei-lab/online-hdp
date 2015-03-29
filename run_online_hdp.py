import sys, os
from corpus import *
import onlinehdp
import cPickle
import random, time
from numpy import cumsum, sum
from itertools import izip
from optparse import OptionParser
from glob import glob
np = onlinehdp.np

def parse_args():
  parser = OptionParser()
  parser.set_defaults(T=300, K=20, D=-1, W=-1, eta=0.01, alpha=1.0, gamma=1.0,
                      kappa=0.5, tau=1.0, batchsize=100, max_time=-1,
                      max_iter=-1, var_converge=0.0001, random_seed=999931111, 
                      corpus_name=None, data_path=None, test_data_path=None, 
                      test_data_path_in_folds=None, directory=None, save_lag=500, pass_ratio=0.5,
                      new_init=False, scale=1.0, adding_noise=False,
                      seq_mode=False, fixed_lag=False)

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
  parser.add_option("--kappa", type="float", dest="kappa",
                    help="learning rate [0.5]")
  parser.add_option("--tau", type="float", dest="tau",
                    help="slow down [1.0]")
  parser.add_option("--batchsize", type="int", dest="batchsize",
                    help="batch size [100]")
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
                    help="testing data prefix for different folds [None], not used anymore")
  parser.add_option("--directory", type="string", dest="directory",
                    help="output directory [None]")
  parser.add_option("--save_lag", type="int", dest="save_lag",
                    help="the minimal saving lag, increasing as save_lag * 2^i, with max i as 10; default 500.")
  parser.add_option("--pass_ratio", type="float", dest="pass_ratio",
                    help="The pass ratio for each split of training data [0.5]")
  parser.add_option("--new_init", action="store_true", dest="new_init",
                    help="use new init or not")
  parser.add_option("--scale", type="float", dest="scale",
                    help="scaling parameter for learning rate [1.0]")
  parser.add_option("--adding_noise", action="store_true", dest="adding_noise",
                    help="adding noise to the first couple of iterations or not")
  parser.add_option("--seq_mode", action="store_true", dest="seq_mode",
                    help="processing the data in the sequential mode")
  parser.add_option("--fixed_lag", action="store_true", dest="fixed_lag",
                    help="fixing a saving lag")
  
  (options, args) = parser.parse_args()
  return options 

def run_online_hdp():
  # Command line options.
  options = parse_args()

  # Set the random seed.
  random.seed(options.random_seed)
  if options.seq_mode:
    train_file = file(options.data_path)
  else:
    train_filenames = glob(options.data_path)
    train_filenames.sort()
    num_train_splits = len(train_filenames)
    # This is used to determine when we reload some another split.
    num_of_doc_each_split = options.D/num_train_splits 
    # Pick a random split to start
    # cur_chosen_split = int(random.random() * num_train_splits)
    cur_chosen_split = 0 # deterministic choice
    cur_train_filename = train_filenames[cur_chosen_split]
    c_train = read_data(cur_train_filename)
  
  if options.test_data_path is not None:
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

  result_directory = "%s/corpus-%s-kappa-%.1f-tau-%.f-batchsize-%d" % (options.directory,
                                                                       options.corpus_name,
                                                                       options.kappa, 
                                                                       options.tau, 
                                                                       options.batchsize)
  print "creating directory %s" % result_directory
  if not os.path.isdir(result_directory):
    os.makedirs(result_directory)

  options_file = file("%s/options.dat" % result_directory, "w")
  for opt, value in options.__dict__.items():
    options_file.write(str(opt) + " " + str(value) + "\n")
  options_file.close()

  print "creating online hdp instance."
  ohdp = onlinehdp.online_hdp(options.T, options.K, options.D, options.W, 
                              options.eta, options.alpha, options.gamma,
                              options.kappa, options.tau, options.scale,
                              options.adding_noise)
  if options.new_init:
    ohdp.new_init(c_train)

  print "setting up counters and log files."

  iter = 0
  save_lag_counter = 0
  total_time = 0.0
  total_doc_count = 0
  split_doc_count = 0
  doc_seen = set()
  log_file = file("%s/log.dat" % result_directory, "w") 
  log_file.write("iteration time doc.count score word.count unseen.score unseen.word.count\n")

  if options.test_data_path is not None:
    test_log_file = file("%s/test-log.dat" % result_directory, "w") 
    test_log_file.write("iteration time doc.count score word.count score.split word.count.split\n")

  print "starting online variational inference."
  while True:
    iter += 1
    if iter % 1000 == 1:
      print "iteration: %09d" % iter
    t0 = time.clock()

    # Sample the documents.
    batchsize = options.batchsize
    if options.seq_mode:
      c = read_stream_data(train_file, batchsize) 
      batchsize = c.num_docs
      if batchsize == 0:
        break
      docs = c.docs
      unseen_ids = range(batchsize)
    else:
      ids = random.sample(range(c_train.num_docs), batchsize)
      docs = [c_train.docs[id] for id in ids]
      # Record the seen docs.
      unseen_ids = set([i for (i, id) in enumerate(ids) if (cur_chosen_split, id) not in doc_seen])
      if len(unseen_ids) != 0:
        doc_seen.update([(cur_chosen_split, id) for id in ids]) 

    total_doc_count += batchsize
    split_doc_count += batchsize

    # Do online inference and evaluate on the fly dataset
    (score, count, unseen_score, unseen_count) = ohdp.process_documents(docs, options.var_converge, unseen_ids)
    total_time += time.clock() - t0
    log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
                    total_doc_count, score, count, unseen_score, unseen_count))
    log_file.flush()

    # Evaluate on the test data: fixed and folds
    if total_doc_count % options.save_lag == 0:
      if not options.fixed_lag and save_lag_counter < 10:
        save_lag_counter += 1
        options.save_lag = options.save_lag * 2

      # Save the model.
      ohdp.save_topics('%s/doc_count-%d.topics' %  (result_directory, total_doc_count))
      cPickle.dump(ohdp, file('%s/doc_count-%d.model' % (result_directory, total_doc_count), 'w'), -1)

      if options.test_data_path is not None:
        print "\tworking on predictions."
        (lda_alpha, lda_beta) = ohdp.hdp_to_lda()
        # prediction on the fixed test in folds
        print "\tworking on fixed test data."
        test_score = 0.0
        test_score_split = 0.0
        c_test_word_count_split = 0
        for doc in c_test.docs:
          (likelihood, gamma) = onlinehdp.lda_e_step(doc, lda_alpha, lda_beta)
          test_score += likelihood
          (likelihood, count, gamma) = onlinehdp.lda_e_step_split(doc, lda_alpha, lda_beta)
          test_score_split += likelihood
          c_test_word_count_split += count

        test_log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
                            total_doc_count, test_score, c_test_word_count, 
                            test_score_split, c_test_word_count_split))
        test_log_file.flush()

    # read another split.
    if not options.seq_mode:
      if split_doc_count > num_of_doc_each_split * options.pass_ratio and num_train_splits > 1:
        print "Loading a new split from the training data"
        split_doc_count = 0
        # cur_chosen_split = int(random.random() * num_train_splits)
        cur_chosen_split = (cur_chosen_split + 1) % num_train_splits
        cur_train_filename = train_filenames[cur_chosen_split]
        c_train = read_data(cur_train_filename)

    if (options.max_iter != -1 and iter > options.max_iter) or (options.max_time !=-1 and total_time > options.max_time):
      break
  log_file.close()

  print "Saving the final model and topics."
  ohdp.save_topics('%s/final.topics' %  result_directory)
  cPickle.dump(ohdp, file('%s/final.model' % result_directory, 'w'), -1)

  if options.seq_mode:
    train_file.close()

  # Makeing final predictions.
  if options.test_data_path is not None:
    (lda_alpha, lda_beta) = ohdp.hdp_to_lda()
    print "\tworking on fixed test data."
    test_score = 0.0
    test_score_split = 0.0
    c_test_word_count_split = 0
    for doc in c_test.docs:
      (likelihood, gamma) = onlinehdp.lda_e_step(doc, lda_alpha, lda_beta)
      test_score += likelihood
      (likelihood, count, gamma) = onlinehdp.lda_e_step_split(doc, lda_alpha, lda_beta)
      test_score_split += likelihood
      c_test_word_count_split += count

    test_log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
                        total_doc_count, test_score, c_test_word_count, 
                        test_score_split, c_test_word_count_split))
    test_log_file.close()

if __name__ == '__main__':
  run_online_hdp()
