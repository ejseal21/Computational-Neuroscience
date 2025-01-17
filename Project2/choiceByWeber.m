function Tj = choiceByWeber(curr_A, w_code, alpha)
%%choiceByWeber Choice-by-Weber coding layer input match function
%
% Parameters
%%%%%%%%%%%%
% curr_A: Current input
% w_code: Coding layer wts for committed nodes only!
% alpha: Choice parameter
[M, C_max] = size(w_code);
M = M/2;
Tj = zeros(1, C_max);
for j = 1:C_max
  Tj(:, j) = (sum(min(curr_A, w_code(:, j))))/(alpha + sum(w_code(:, j)));
end

end