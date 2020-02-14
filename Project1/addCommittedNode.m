function [C, w_code, w_out] = addCommittedNode(C, curr_A, K, w_code, w_out)
  %%addCommittedNode commit a new coding unit to code the current input pattern. This involves setting the wts of the newly
  %%committed coding unit to the current input pattern, and setting the wts going from the newly committed coding unit C
  %%to the output layer unit coding the correct output class (K) equal to 1.
  %
  % Logic: If the current input were presented to the network again (immediately after),
  % the new committed unit would activate and then active the output unit that represents the correct class. 
  % Then ARTMAP would get the classification correct.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % C: int. Current number of committed coding units (before the addition that this function makes).
  % curr_A: matrix. size=(2*M, 1). Current data sample, complement coded.
  % K: int. Class index of curr_A. 1,2,...
  % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
  % w_out: matrix. size=(C_max, n_classes). Coding-layer-to-output-class-layer adaptive weights.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % C: int. Current number of committed coding units (after the addition that this function makes).
  % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
  % w_out: matrix. size=(C_max, n_classes). Coding-layer-to-output-class-layer adaptive weights.
end