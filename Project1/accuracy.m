function acc = accuracy(pred_y, true_y)
  %%accuracy computes the accuracy between [0, 1] between int coded predicted class labels `pred_y` and true labels
  %%`true_y`.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % pred_y: matrix. size=(1), #samples (N)). Predicted classes of data samples represented as ints 1, 2..., #classes.
  % true_y: matrix. size=(1), #samples (N)). Actual classes of data samples represented as ints 1, 2..., #classes.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % acc: double. Accuracy on the dataset.
  
   indices =  find(pred_y~=true_y);
   acc = 1-(size(indices)/size(true_y));
end