function [ x_pred,P_pred ] = kf_predict(x, P, F, Q, u, B, Qu)
%KF_PREDICT Perform the discrete-time KF state prediction 
% step, under the assumption of additive process noise.
%
% Parameters
% ----------
% x: column vector
%   The (xDim x 1) state estimate at the previous time-step.
% P: matrix
%   The (xDim x xDim) state covariance matrix at the previous
%   time-step.
% F: matrix
%   An (xDim x xDim) state transition matrix.
% Q: matrix
%   The (xDim x xDim) process noise covariance matrix.
% u: column vector, optional
%   An optional (uDim x 1) control input.
%   If omitted, no control input is used.
% B: matrix, optional
%   An optional (uDim x uDim) control gain matrix.
%   If omitted, B is assumed to be 1.
% Qu: matrix, optional
%   An optional (uDim x uDim) control noise covariance
%   matrix. If omitted, Q is assumed to be 0.
%
% Returns
% -------
% x_pred: column vector
%   The (xDim x 1) predicted state estimate.
% P_pred: matrix
%   The (xDim x xDim) predicted state covariance matrix.

    x_pred = F*x + B*u;
    P_pred = F*P*F' + Q + B*Qu*B';

end

