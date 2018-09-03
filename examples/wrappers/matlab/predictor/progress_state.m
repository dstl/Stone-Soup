function [ x_k1,P_k1 ] = progress_state( x_k, P_k, u_k, F, B, Q)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    x_k1 = F*x_k + B*u_k;
    P_k1 = F*P_k*F' + Q;


end

