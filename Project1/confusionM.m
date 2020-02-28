function confusion = confusionM(pred_y, true_y, n_classes)
%%Computes the confusion matrix
%
% Parameters:
%%%%%%%%%%%%%%%%%%%%
% pred_y: matrix. size=(1), #samples (N)). Predicted classes of data samples represented as ints 1, 2..., #classes.
% true_y: matrix. size=(1), #samples (N)). Actual classes of data samples represented as ints 1, 2..., #classes.
%
% Returns:
% confusion matrix. size = (2, 2)
%%%%%%%%%%%%%%%%%%%%

N = numel(pred_y);
confusion = zeros(n_classes, n_classes);
for i = 1:N
  a = true_y(i);
  p = pred_y(i);
  confusion(a, p) =  confusion(a, p) + 1;

end