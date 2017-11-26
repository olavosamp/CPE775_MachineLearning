cov = [1.2 0.4; 0.4 1.8];
u1 = [ 0.1; 0.1];
u2 = [ 2.1; 1.9];
u3 = [-1.5; 2.0];

x  = [ 1.6; 1.5];
x2 = [2.1; 1.9];

g1 = discriminantFunction(x, u1, cov)
mh1 = mahalanobis(x2, u1, cov)

g2 = discriminantFunction(x, u2, cov)
mh2 = mahalanobis(x2, u2, cov)

g3 = discriminantFunction(x, u3, cov)
mh3 = mahalanobis(x2, u3, cov)