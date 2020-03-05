function [C, w_code] = fuzzy_art_train(data, verbose, show_plot, varargin)
  %%fuzzy_art fuzzy ART unsupervised pattern learning algorithm
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % data: matrix. size=(#dimensions (M), #samples (N)). Data samples normalized in the range [0,1].
  % verbose: boolean. If false, suppresses ALL print outs.
  % show_plot: boolean. If true, show and update the category box plot during each iteration of the epoch.
  %   Only applies to CIS dataset.
  % varargin: cell array. variable length optional parameters.
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % C: int. Number of committed cells in the coding layer (y cells).
  % w_code: array. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
  %   This tells us how to activate committed coding cells based on a new input pattern.
  %
  % NOTE: We need both sets of learned weights w_code, w_out to form a prediction for a test input, which is why we're
  % returning them.
  % NOTE: C tells us which weights in w_code and w_out are currently used/relevant.
  
  % Set parameters
  %
  % Coding layer y choice parameter ("tie breaker" for activation values). (0, Inf)
  alpha = 0.01;
  % Learning rate. [0, 1]. 1 means fast one-shot learning
  beta = 1.0;

  % Baseline vigilance / matching criterion. [0, 1]. 0 maximizes code compression.
  p = 0.75;
  % Number of training epochs. We only need 1 when beta=1
  n_epochs = 1;
  % Max number of commitable coding cells. C_max start uncommitted.
  C_max = size(data, 2);
 
  % Override default settings/parameters
  for arg = 1:2:length(varargin)
    switch varargin{arg}
      case 'alpha'
        alpha = varargin{arg+1};
      case 'beta'
        beta = varargin{arg+1};
      case 'p'
        p = varargin{arg+1};
      case 'n_epochs'
        n_epochs = varargin{arg+1};
      case 'C_max'
        C_max = varargin{arg+1};
    end
  end
  


[M, N] = size(data_x);
% initialize weights
w_code = ones(2*M, C_max);
w_out = zeros(C_max, n_classes);
%complement code input samples
A = complementCode(data_x);

% loop for training epochs
for num_e = 1: n_epochs
  % iterate thru samples
  for i = 1:N
    
    p = p_base;
    Tj = choiceByWeber(A(:, i), w_code, alpha);
    
    [pm_inds, pm_sorted_inds] = possibleMatchInds(Tj, alpha, M);
%     sort(Tj, 'Ascending');
    pass = 0;
    for c = 1:n_above_thre
      % vigilance test
      if sum(min(A(:,i), w_code(:,pm_sorted_inds(c))))/M >= p
        if data_y(i) == find(w_out(pm_sorted_inds(c), :)==1)  
          w_code = updateWts(beta, A(:, i), w_code, pm_sorted_inds(c));
          pass = 1;
          break
%         else  
%           p = matchTracking(A(:, i), w_code, pm_sorted_inds(c), M, e);
        end
      end
    end
    
    if C_max > C  % confused
      if pass == 0
        [C, w_code, w_out] = addCommittedNode(C, A(:, i), data_y(i), w_code, w_out);
      end
    end
    
  end
  if show_plot == 1
    plotCategoryBoxes(A, data_y, i, C, w_code, w_out, "train");
  end
end



end