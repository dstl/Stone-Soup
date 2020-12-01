function measHist = getMeasHist(hypotheses, slideWindow)

% Get measHist(end-i+1, h) = measurement assigned to scan i-1 timesteps ago
% for hypothesis h (NaN if none)
numHypotheses = numel(hypotheses);
measHist = zeros(slideWindow, numHypotheses);
for i = 1:numHypotheses
    thislen = size(hypotheses(i).measHistory,1);
    if thislen < slideWindow
        measHist(:,i) = [zeros(slideWindow - thislen, 1); hypotheses(i).measHistory(:,2)];
    else
        measHist(:,i) = hypotheses(i).measHistory(end-slideWindow+1:end,2);
    end
end

% hypotheses.measHistory
% numHypotheses = numel(hypotheses);
% measHist = zeros(slideWindow, numHypotheses);
% for i = 1:numHypotheses
%     thislen = size(hypotheses(i).measHistory,2);
%     if thislen < slideWindow
%         measHist(:,i) = [zeros(slideWindow - thislen, 1); hypotheses(i).measHistory'];
%     else
%         measHist(:,i) = hypotheses(i).measHistory(end-slideWindow+1:end)';
%     end
% end
