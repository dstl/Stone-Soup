function [bestHypotheses, hypothesesOut] = mfa(hypotheses, slideWindow)

% Get max track ID and measurement history over the sliding window
maxTrackID = max([hypotheses.trackID]);
measHist = getMeasHist(hypotheses, slideWindow);
currentScanNum = hypotheses(1).measHistory(end,1);
windowScans = currentScanNum - (slideWindow-1:-1:0)';

% Empty hypothesis struct with same field names for preallocation
fields = fieldnames(hypotheses);
emptyhyp = cell2struct(cell(length(fields),1), fields);

% Add dummy tracks for each non-null measurement - each has a hypothesis
% that the measurement was used and that it wasn't so that all of the
% measurement hypotheses are used by the assignment
% (find better way to handle constraint being == rather than <=?)
dummyHypotheses = cell(slideWindow, 1);
dummyTrackID = maxTrackID + 1;
for k = 1:slideWindow
    idx = (measHist(end-k+1,:)~=0) & ~isnan(measHist(end-k+1,:));
    thesemeas = unique(measHist(end-k+1, idx));
    dummyHypotheses{k} = repmat(emptyhyp, 2*numel(thesemeas), 1);
    for j = 1:numel(thesemeas)
        % Null hypothesis
        dummyHypotheses{k}(2*j-1,1).cost = 0;
        dummyHypotheses{k}(2*j-1,1).trackID = dummyTrackID;
        dummyHypotheses{k}(2*j-1,1).measHistory = [windowScans zeros(slideWindow,1)];
        % Measurement hypothesis
        dummyHypotheses{k}(2*j,1).cost = 0;
        dummyHypotheses{k}(2*j,1).trackID = dummyTrackID;
        dummyHypotheses{k}(2*j,1).measHistory = [windowScans zeros(slideWindow,1)];
        dummyHypotheses{k}(2*j,1).measHistory(end-k+1,2) = thesemeas(j);
        dummyTrackID = dummyTrackID + 1;
    end
end
dummyHypotheses = cat(1, dummyHypotheses{:});

% Run MFA
[bestHypotheses, hypothesesOut] = dataAssocPRH([hypotheses; dummyHypotheses],...
    slideWindow);

% Delete dummy tracks
bestHypotheses = bestHypotheses([bestHypotheses.trackID] <= maxTrackID);
hypothesesOut = hypothesesOut([hypothesesOut.trackID] <= maxTrackID);
