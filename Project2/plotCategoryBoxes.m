function plotCategoryBoxes(A, n, C, w_code, train_or_test, y_pred)
%plotCategoryBoxes plots the data and the category boxes to visualize fuzzy ARTMAP.
%
% Goal: Plot the data and network state on a SINGLE figure/plot. In other words, this function should only generate
% ONE pop-up window, even if this function is called multiple times.
% If called a second time, the previous plot should be wiped clean and your function should draw everything fresh
% to reflect the current data sample and network state. The intended use case is to call it in every iteration of the
% main training/testing loop to visualize what's happening.
% NOTE: Only intended for use on the CIS dataset!
%
% Parameters
%%%%%%%%%%%%
% A: matrix. size=(2*M, N). All the data samples / inputs (all N of them)
% m: matrix. size=(1, N). Class of each data sample
% n: int. Index of current input
% M: Number of features (before complement coding)
% C: Number of committed nodes
% w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights. Only 1 thru C are committed.
% w_out: matrix. size=(C_max, n_classes). Coding-layer-to-output-class-layer adaptive weights.
%   This tells us which output class each committed coding cell is associated with (which class it tends to predict).
% train_or_test. string. 'train' if plotting during training, 'test' if plotting during prediction.
% y_pred: matrix. size=(n_classes, # test samples). NOTE: Only provided when processing test set/making predictions.
%   Each column vector contains the prediction for the corresponding test sample.
%   In fast learning this is a matrix of one-hot coded vectors: every vector contains a 1 in the predicted
%   class, 0s elsewhere.
%
% TODO
%%%%%%%%%%%%
% 1) Create a new figure with an equal/square aspect ratio (x scale == y scale). You'll know if this isn't working
% if your circle looks like an ellipse!
% 2) Draw a circle centered at (x, y) = (0.5, 0.5) with radius 1/sqrt(2*pi). This matches the "ground truth" boundary
% between circle and square class points. Set your x/y limits appropriately, knowing that the CIS data always falls in
% the unit square. You shouldn't make it hard to see samples that fall on the extreme borders.
% 3) Draw the category boxes for all COMMITTED coding units color-coded by class (circle->blue, square->red).
% For each coding unit, trace out a rectangle by connecting the following path:
%   x: [w(1), 1-w(3), 1-w(3), w(1), w(1)]
%   y: [w(2), w(2), 1-w(4),1-w(4), w(2)]
% where w is shorthand for the w_code and the indices are COLUMN indices.
% NOTE: You cannot simply copy-paste the above.
%
% If training:
% 4) Draw the current data sample being processed in black with a + shaped marker. Show the actual sample, not the
% complement coded version.
% 5) Draw all the previous data samples color coded by their class (circle->blue, square->red). To help visual
% discriminability, make the circle sample markers a circle shape, and the square sample markers square shaped.
%
% If testing/making prediction:
% 4) Draw all predicted up to and including the current data sample, color-coded by the PREDICTED class.
% 5) If the predicted class is correct for a data sample, make the plot marker a circle. if the predicted class is
% incorrect, make the marker an x.
%
% Tip: It is possible to write this function without any loops

figure(1);
clf;
pbaspect([1 1 1]);
hold on;
viscircles([.5 .5],1/sqrt(2*pi), 'Color', 'c');

plot(A(1,n),A(2,n), '+', 'Color','k', 'LineWidth', 2);
for index = 1:n-1
      plot(A(1,index),A(2,index), 'o', 'Color', 'b', 'LineWidth', 2)
end
  
for row = 1:C
  x = [w_code(1, row), 1-w_code(3, row), 1-w_code(3, row), w_code(1, row), w_code(1, row)];
  y = [w_code(2, row), w_code(2, row), 1-w_code(4, row),1-w_code(4, row), w_code(2, row)];
  plot(x,y, 'LineWidth', 2, 'Color', 'b')
end
hold off;

end