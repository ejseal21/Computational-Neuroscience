function p = matchTracking(curr_A, w_code, active_ind, M, e)
  %%matchTracking adjusts the vigilance (p) so that candidate coding cells in the ART search cycle will be more
  %%selective / less tolerant of matching (resonating) with the current sample.
  %%See Live Script for refresher on match tracking equation.
  %
  % Parameters
  %%%%%%%%%%%%
  % curr_A. matrix. size=(2*M, 1). Current input
  % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
  % active_ind: int. Index of the currently active coding cell.
  % M: int. Number of features in the input pattern (before complement coding).
  % e: double. Match tracking constant. Amount to temporarily adjust the vigilance compared to the current degree of
  % match between the data sample and coding unit J weights.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % p: double. Adjusted vigilance parameter.
end