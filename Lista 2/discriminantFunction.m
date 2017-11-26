function [result] = discriminantFunction(x, u, cov)

result = (cov\u)'*x - 0.5*u'*(cov\u);
end