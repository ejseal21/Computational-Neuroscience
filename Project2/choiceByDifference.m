function Tj = choiceByDifference(curr_A, w_code, C, alpha, M)
  %%choiceByDifference choice-by-difference coding layer net input function
  %
  % Computes 'fuzzy AND' between input pattern and each committed unit's wts. 
  % Applies the choice-by-difference function to get netIn for each of the C committed coding units 
  % (see Live Script for refresher on equation).
  %
  % Parameters
  %%%%%%%%%%%%
  % curr_A: matrix. size=(2*M, 1). Current input
  % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
  % C: int. Number of committed nodes
  % alpha: double. Choice/regularization parameter
  % M: int. Number of features (before complement coding)
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % Tj: matrix. size=(1, C). Net input for committed units in the coding layer.
  
  Tj = zeros(1, C);
  for j = 1:C
      left = sum(min(curr_A, w_code(:,j)));
      right = (1 - alpha) * (M - sum(w_code(:,j))); 
      Tj(:, j) = left + right;
  end
end