function [final_preds, acc] = artmap_ensemble(train_x, train_y, test_x, test_y, n_classes, n_voters, verbose)
  %%artmap_ensemble train an ensemble of fuzzy ARTMAP systems and classify samples according to majority vote.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % train_x: matrix. size=(#dimensions (M), #samples (N1)). Training data samples normalized in the range [0,1].
  % train_y: matrix. size=(1, N1). Classes of training samples represented as ints 1, 2..., #classes.
  % test_x: matrix. size=(#dimensions (M), #samples (N2)). Test data samples normalized in the range [0,1].
  % test_y: matrix. size=(1, N2). Classes of test samples represented as ints 1, 2..., #classes.
  % n_classes: int. Number of classes in the dataset. This is a parameter because not all class values may be in the
  %   training set. For example, our training set may have 4 classes, and the test set has 5 (one missing).
  % n_voters: int. Number of fuzzy ARTMAP systems to train and participate in the ensemble voting.
  % verbose: boolean. If false, suppresses ALL print outs.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % final_preds: matrix. size=(1, N2). Final class predictions. Contains ints. e.g. [1, 2, 2, 1, ...]
  % acc: double. Accuracy on the test set.
  %
  % TODO:
  % For each of the ARTMAP systems:
  % 1) Shuffle the training data and classes.
  % 2) Train each system on the shuffled data.
  % 3) Test on the trained weights to get the system's vote vectors for each test sample.
  % 4) Tally (sum) all the votes obtained across the voters, then find the class indices that achieve the majority (max)
  % vote. Break ties by selecting the 1st class.
  % 5) Compute the accuracy and print out the class predictions compared to the true classes of the data.

  % Set random seed for reproduceability
  rng(0);
  
  % Set parameters
  %
  % Coding layer y choice parameter. (0, 1)
  alpha = 0.01;
  % Learning rate. [0, 1]. 1 means fast one-shot learning
  beta = 1;
  % Matching tracking update rate. (-1, 1)
  e = -0.001;
  % Baseline vigilance / matching criterion. [0, 1]. 0 maximizes code compression.
  p_base = 0;
  % Number of training epochs
  n_epochs = 1;
  % Max number of commitable coding cells. All C_max cells start uncommitted.
  C_max = 20;
  
  tally = zeros(n_classes, size(test_x, 2), n_voters);
  C = 0;
  w_code = [];
  w_out = [];
  yh_pred = zeros(n_voters, size(test_x, 2));
  for voter=1:n_voters
      r = randperm(size(train_y, 2));
      shuf_x = train_x(:, r);
      shuf_y = train_y(:, r);
      [C, w_code, w_out] = artmap_train(shuf_x, shuf_y, 2, 0, 0);
      tally(:, :, voter) = artmap_test_wta(C, w_code, w_out, test_x, test_y, 2, 0, 0);
  end
  
  tally = sum(tally, 3);
  [useless, final_preds] = max(tally, [], 1);
  acc = accuracy(final_preds, test_y);
  
end