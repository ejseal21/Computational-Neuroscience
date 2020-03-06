function c_pred = fuzzy_art_predict(C, w_code, data, verbose, varargin)
  %%Fuzzy ART Predict: Return the index of the coding cell that activates the most to each data sample.
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % C: int. Number of committed cells in the coding layer (y cells).
  % w_code: matrix. size=(2*M, C_max) Learned input-to-coding-layer adaptive weights. 
  % data: matrix. size=(#dimensions (M) x #samples (N)). Test Dataset. Values normalized in the range [0,1].
  % verbose: boolean. If false, suppresses ALL print outs.
  % varargin: cell array. variable length optional parameters.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % c_pred. matrix. size=(# test samples, 1). Column vector contains the index of the activated coding layer cell 
  % in response to each test sample.

  
  % Set parameters
  %
  % Coding layer y choice parameter. (0, Inf)
  alpha = 0.01;
  
  % Override default settings/parameters
  for arg = 1:2:length(varargin)
    switch varargin{arg}
      case 'alpha'
        alpha = varargin{arg+1};
    end
  end
  A = complementCode(data);
  
  for i = 1:C
    Tj = choiceByWeber(A(:, i), w_code, alpha);
    find(Tj == max(Tj));
  end
  if show_plot == 1
      plotCategoryBoxes(A, data_y, i, C, w_code, w_out, "test", yh_pred);
  end



end