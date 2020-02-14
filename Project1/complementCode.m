function Ac = complementCode(data)
  %%complementCode complement codes data. Size goes from (M, N) -> (2M, N)
  %
  % Parameters:
  %%%%%%%%%%%%%%%%%%%%
  % data: array. size = (#dimensions (M), #samples (N)). Data samples normalized in the range [0,1].
  %
  %
  % Returns:
  %%%%%%%%%%%%%%%%%%%%
  % Ac: array. Values in the range [0,1]. size = (2*#dimensions (M), #samples (N)).
  %   1st half along 1st dimension is `data`. 2nd half along dimension is 1-`data`.
end