function pm_sorted_inds = possibleMatchInds(Tj, alpha, M)
  %%possibleMatchInds returns the indices of sorted above threshold possible category matches.
  %%i.e.: apply the following threshold-linear netAct function then sort candidates max->min.
  %%See Live Script for refresher on possible match threshold equation.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % Tj: matrix. size=(1, C). Committed unit net inputs (e.g. determined by choice-by-difference).
  % alpha: double. Choice/regularization parameter
  % M: int. Number of features (before complement coding)
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % pm_inds. matrix. size=(1, C*). int indices of above-thresold Tj values. C* is the number of above threshold Tj values.
  % pm_sorted_inds. matrix. size=(1, C*). int indices of max-to-min sorted above-thresold Tj values.
  
  %pm_inds = find(Tj > alpha * M);
  sort_Tj = sort(Tj, 'descend');
  for i = 1:numel(Tj)                      %should probably try to get rid of this loop but dont know how
    pm_sorted_inds(1, i) = find(Tj == sort_Tj(i));
  end
end

