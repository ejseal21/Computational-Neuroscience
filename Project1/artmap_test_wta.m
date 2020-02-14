function yh_pred = artmap_test_wta(C, w_code, w_out, data_x, data_y, n_classes, verbose, show_plot, varargin)
  %%ARTMAP Default ARTMAP implementation of the Fuzzy ARTMAP classifer with winner-take-all testing.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % C: int. Number of committed cells in the coding layer (y cells).
  % w_code: matrix. size=(2*M, C_max) Learned input-to-coding-layer adaptive weights. 
  % w_out: matrix. size=(C_max, n_classes). Learned coding-layer-to-output-class-layer adaptive weights.
  % data_x: matrix. size=(#dimensions (M) x #samples (N)). Test Dataset. Values normalized in the range [0,1].
  % data_y: matrix of ints. size=(1, N). Classes represented as ints 1, 2..., #classes.
  %   NOTE: Only used for plotting purposes. Should not be used in algorithm!
  % n_classes: int. Number of classes in the dataset. This is a parameter because not all class values may be in the
  %   test set. For example, our test set may have 4 classes, and the training set has 5 (one missing).
  % verbose: boolean. If false, suppresses ALL print outs.
  % show_plot: boolean. If true, show the category box plot during each iteration of the epoch.
  % varargin: cell array. variable length optional parameters.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % yh_pred. matrix. size=(n_classes, # test samples). Each column vector contains the prediction for the corresponding
  % test sample. In fast learning this is a matrix of one-hot coded vectors: every vector contains a 1 in the predicted
  % class, 0s elsewhere.

  
  % Set parameters
  %
  % Coding layer y choice parameter. (0, 1)
  alpha = 0.01;
end