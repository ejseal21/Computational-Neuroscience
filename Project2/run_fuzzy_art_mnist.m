function [mnist_test_y, code_inds, C] =  run_fuzzy_art_mnist(mnist_path, sets, ...
    num_exemplars, num_classes, noisify_test, erase_test, plot_wts, plot_recall, varargin)
  %%run_fuzzy_art_mnist Trains fuzzy ART on the MNIST dataset and recover "memories" prompted by the test set using the
  %%Fuzzy ART predict function. A memory refers to the weights for the winning code unit for each test data sample.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % mnist_path: str. Relative path to MNIST dataset.
  % sets: cell array of strings. size=(2, 1). Entries contain either 'train' or 'test'. 1st entry is which MNIST set is
  %   used during training. 2nd entry is MNIST set used during prediction. For example, {'test', 'test'} would mean use
  %   the test set for both training and prediction. If 'train', load in MNIST training set. Otherwise, load test set.
  % num_exemplars: matrix of ints. size=(2, 1).  Each number means how many digits should select we from each class from
  %   the dataset? For example, if num_exemplars = 5, load the 1st 5 images of 0s in the dataset and also the 1st 5 1s,
  %   also the 1st 5 2s...etc. 
  %   1st entry is how many exemplars should be loaded for training. 2nd entry is for how many load in prediction
  %   (test). For example, [5, 1] means load 5 samples of each digit from MNIST during training and 1 per digit during
  %   prediction/testing.
  % num_classes: int. How many digit classes are we going to load? For example if = 10, load digits of all types (0-9).
  %   If = 3, only load 0s, 1s, and 2s. You can assume for simplicity that selections are made in-order starting from 0.
  % noisify_test: boolean. Do we add noise to the test images? Ignore/set to false until the notebook instructions state
  %   otherwise.
  % erase_test: boolean. Do we erase part of each test image? Ignore/set to false until the notebook instructions state
  %   otherwise.  
  % plot_wts: boolean. If true, make a square grid plot showing images of all the learned weights of each of the committed
  % coding units. Reshaping to 28x28 pixels will be necessary.
  % plot_recall: boolean. If true, make a 2 column figure.
  %   Each row in left column shows each input test set image.
  %   Each row in right column shows the corresponding weight (memory) of the coding unit that was most active when you
  %   presented the test image to Fuzzy ART (determined in predict function).
  % varargin: cell array. size=variable. List of strings and values to pass along to fuzzy_art_train (and be parsed
  % there). Holds things like hyperparameters (e.g. p, alpha, beta, num_epochs, etc.).
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % mnist_test_y. matrix of doubles. size=(num_test_exemplars*num_classes, 1).
  %   Int-codes of MNIST class for each test sample. Codes should be ints in the range (inclusive) [0, 9] and literally
  % mean which digit is shown in the image.
  % code_inds. matrix of ints. size=(num_test_exemplars*num_classes, 1).
  %   Indices of the max active coding units to each test image.
  % C. int. Number of committed coding layer units.
  %
  % TODO: (also see notebook)
  % - Load MNIST train and test sets.
  % - Fuzzy ART on training set.
  % - Possibly visualize the weights.
  % - Predict the categories of the test set images.
  % - Possibly visualize the weights of the most active coding units during prediction to the test images.
  
  train = load_mnist('data/MNIST', 'train', num_exemplars(1), num_classes, 0);
  test = load_mnist('data/MNIST', 'test', num_exemplars(2), num_classes, 0);
  
  [C, w_code] = fuzzy_art_train(train, 1, 1, varargin{:});
  

  c_pred = fuzzy_art_predict(C, w_code, test, 1);
  
  disp(c_pred(:, 1))
  
  if plot_wts
    for class = 1:C
        subplot(ceil(sqrt(C)),ceil(sqrt(C)), class), imshow(reshape(w_code(1:784 ,class), [28 28])');
    end 
  end
  
  if plot_recall
      
  end
  
  
  
end







