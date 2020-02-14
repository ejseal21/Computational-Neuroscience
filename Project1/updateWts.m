function w_code = updateWts(beta, curr_A, w_code, active_ind)
  %%updateWts updates the wts to the active coding unit in `w_code` at index `active_ind`.
  % NOTE: Only updates the active coding unit's wts. The rest remain unchanged.
  % When we have beta=1 (fast learning) the active unit's wts become equal to the 'fuzzy AND' of curr wts and the curr
  % input.
  %
  % Parameters
  %%%%%%%%%%%%
  % beta: double. Wt learning rate
  % curr_A. matrix. size=(2*M, 1). Current input
  % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
  % active_ind: int. Index of the currently active coding layer cell.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights. Wt updated only for the cell at active_ind.
end