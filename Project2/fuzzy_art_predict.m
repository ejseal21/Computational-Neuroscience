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
  [M, N] = size(data);
  
  % Override default settings/parameters
  for arg = 1:2:length(varargin)
    switch varargin{arg}
      case 'alpha'
        alpha = varargin{arg+1};
    end
  end
  
  A = complementCode(data);
  c_pred = zeros(N, 1);
  
  for samp = 1:N
      Tj = choiceByWeber(A(:, samp), w_code(:, 1:C), alpha);  % not sure about w_code(:, 1:C)
      c_preds = find(Tj == max(Tj));
      c_pred(samp) = c_preds(1);
  end




end