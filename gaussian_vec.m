function [y] = gaussian_vec(x, varargin)
  mu = 0;
  sig2 = 1;
  try
    mu = varargin{1};
    sig2 = varargin{2};
  end
  y = 1/(sqrt(sig2*2*pi))*exp(-1/2*square(x-mu)/sig2)
end

%fx1 = @(x, mu, sig2) 1/(sqrt(sig2*2*pi))*exp(-1/2*square(x-mu)/sig2);