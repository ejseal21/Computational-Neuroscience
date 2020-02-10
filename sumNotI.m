function [x] = sumNotI(y)
  s = sum(y(:));
%    for i = 1:numel(y)
%      x(i) = s - y(i);
%    end
  m = linspace(1,numel(y), numel(y));
  m = reshape(m, size(y));
  x = s - y(m);
end