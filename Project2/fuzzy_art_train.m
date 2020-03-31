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
  fast = 0;
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
      case 'fast'
        fast = varargin{arg+1};
    end
  end
  

[M, N] = size(data);
% initialize weights
w_code = ones(2*M, C_max);
%complement code input samples
A = complementCode(data);
C = 0;

% loop for training epochs
for num_e = 1: n_epochs
  % iterate thru samples
  for i = 1:N
    
    Tj = choiceByWeber(A(:, i), w_code, alpha);
    sort_Tj = sort(Tj, 'descend');

    for c = 1:numel(Tj)
      % vigilance test
      ind = find(Tj==sort_Tj(c));
      if sum(min(A(:,i), w_code(:,ind)))/M >= p
          if ind <= C
            if fast{1}
              w_code = updateWts(beta, A(:, i), w_code, ind);
            else
              w_code = updateWts(1, A(:, i), w_code, ind);
            end
            break
          else
            [C, w_code] = addCommittedNode(C, A(:, i), w_code);
            break
          end
      end
    end
    
  end
  
  if show_plot == 1
    plotCategoryBoxes(A, i, C, w_code, "train");
  end
end

end