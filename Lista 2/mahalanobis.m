function [ result ] = mahalanobis( x, u, cov )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

result = (x - u)'*(cov\(x - u));
end

