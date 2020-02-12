function [y] = gaussian_vec(x, varargin)
%  mu = 0;
%  sig2 = 1;
  try
    mu = varargin{1};
  catch
     mu = 0;
  end
  try
    sig2 = varargin{2};
  catch
    sig2 = 1;
  end
  
  y = 1/(sqrt(sig2*2*pi))*exp(-1/2*(x-mu).^2/sig2);
end

%fx1 = @(x, mu, sig2) 1/(sqrt(sig2*2*pi))*exp(-1/2*square(x-mu)/sig2);