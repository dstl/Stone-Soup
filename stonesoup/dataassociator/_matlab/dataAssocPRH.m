function [bestHypotheses, hypothesesOut] = dataAssocPRH(hypotheses, slideWindow)

% (look at adding inequality constraint?)
% Dummy tracks assumed added so that every measurement is used by exactly
% one track

% Terminate if gap between primal and dual costs is less than this
gap_threshold = 0.02;
% Initialise maximum num of iterations
maxIteration = 1e1;

% branch and bound options
options = optimoptions('intlinprog','Display','off');

% num of single target hypotheses
numHypotheses = length(hypotheses);        

% Get track IDs and how many tracks there are
hypothesisTrackIDs = [hypotheses.trackID];
uniqueTrackIDs = unique(hypothesisTrackIDs, 'stable');
numTracks = length(uniqueTrackIDs);

% cost of single target hypotheses
hypothesisCosts = [hypotheses.cost]';

% calculate the length of trajectory of each single target hypothesis in
% tracks, used for determine the num of scans should be used
hypothesisHistLen = arrayfun(@(x)size(x.measHistory, 1), hypotheses);
% ensure we constrain the sliding window to the maximum history length
maxHypothesisHistLen = max(hypothesisHistLen); % maxtralen-scan data association
maxHypothesisHistLen = double(min(maxHypothesisHistLen, slideWindow));

% construct binary indicator matrix for constraint
% (1): each track should only be used once; this constraint is neccessary
% for the implementation of dual decomposition, since sliding window is used.
% (2): each measurement in each scan should only be used once
% track_index{i} = hypotheses corresponding to target i
% measTrack_index{k}{j, i} = hypothesis indices which assign
%   measurement j to track i in scan k
% trackNull_index{k, i} = hypothesis indices which assign the null
%   hypothesis to track i in scan k
[A0, A, track_index, measTrack_index, trackNull_index] = getConstraints(...
    hypotheses, maxHypothesisHistLen);
Amatrix = [A0; cat(1, A{:})];

% dual decomposition, solution is a binary indicator vector, decides which
% single target hypotheses are included in the "best" global hypotheses.

% subproblem t: min(c/t+\deltat)*u, s.t. [A0;At]*u = [b0;bt];

% Larange multiplier \delta is initialised with 0
delta = zeros(numHypotheses,maxHypothesisHistLen);
% subproblem solutions
u_hat = false(numHypotheses,maxHypothesisHistLen);

hasRepeated = false; % Terminate the loop if the same dual solution repeats
% store the best feasible primal cost obtained so far (upper bound)
bestPrimalCost = inf;
uprimal = false(numHypotheses,1);

numIteration = 0;
while (numIteration<maxIteration && ~hasRepeated)
    
    % get suboptimal solution for each subproblem
    subDualCost = zeros(maxHypothesisHistLen, 1);
    for k = 1:maxHypothesisHistLen
        
        % get hypothesis costs including Lagrangians
        c_hat = hypothesisCosts/maxHypothesisHistLen + delta(:,k);

        [cost, idxCost, nullCost, idxNullCost] = get2dCostMatrix(...
            c_hat, measTrack_index{k}, trackNull_index(k,:));
        assignments = assignmentoptimalNull(cost, nullCost);
        % Assign hypothesis indicators associated with chosen assignments
        % to true
        assignedHypotheses = false(numHypotheses,1);
        for i = 1:numTracks
            if assignments(i)>0
                assignedHypotheses(idxCost(i,assignments(i))) = true;
            else
                assignedHypotheses(idxNullCost(i)) = true;
            end
        end

        u_hat(:,k) = assignedHypotheses;
        subDualCost(k) = c_hat'*u_hat(:,k);
    end
    
    % Get proportion of assignments over the measurement scans for each
    % hypothesis - if mean is zero or one then all scans agree
    u_hat_mean = sum(u_hat,2)/maxHypothesisHistLen;
    
    % All the subproblem solutions are equal means we have found the
    % optimal solution
    if all(u_hat_mean==1 | u_hat_mean==0)
        uprimal = u_hat(:,1);
        break;
    end

    % calculate dual cost
    dualCostHat = sum(subDualCost);
    
    % Get primal solution
    [uprimalHat, primalCostHat] = getPrimalSolution(u_hat_mean, Amatrix,...
        hypothesisCosts, options);
    
    % Replace best primal cost
    if primalCostHat < bestPrimalCost
        bestPrimalCost = primalCostHat;
        uprimal = uprimalHat;
        hasRepeated = false;
    else
        % jump out the loop if the best primal cost obtained does not increase
        hasRepeated = true;
    end
    
    % jump out the loop if the gap is too small
    gap = (bestPrimalCost - dualCostHat)/bestPrimalCost;
    if gap < gap_threshold
        break;
    end
    
    % calculate step size used in subgradient methods
    % calculate subgradient
    g = u_hat - u_hat_mean;
    % calculate step size used in subgradient method
    stepSize = (bestPrimalCost - dualCostHat)/(norm(g)^2);
    % update Lagrange multiplier
    delta = delta + stepSize*g;
    
    % increase index of iterations
    numIteration = numIteration+1;
end

u = uprimal;

% single target hypotheses in the ML global association hypotheses updating
% pre-existing tracks
I = u(1:numHypotheses)==1;
bestHypotheses = hypotheses(I);

hypothesesOut = pruneHypotheses(hypotheses, bestHypotheses, track_index,...
    slideWindow);

%--------------------------------------------------------------------------

function hypothesesOut = pruneHypotheses(hypotheses, bestHypotheses,...
    track_index, slideWindow)

% N-scan pruning
numHypotheses = numel(hypotheses);
idx_remain = false(numHypotheses,1);
% Length of bestHypotheses = number of tracks?
for i = 1:length(bestHypotheses)
    trajectoryLength = size(bestHypotheses(i).measHistory,1);
    %if trajectoryLength>=nc && numHypsPerTarget(i)==1 && bestHypotheses(i).r==0
    %    % prune tracks with only null-hypothesis with length no less than nc
    %else
    if trajectoryLength >= slideWindow
        % Hypotheses of bestHypothesis at start of sliding window
        thisHyp = bestHypotheses(i).measHistory(1:end-slideWindow+1, 2);
        for j = track_index{i}
            % Keep the hypotheses for this track if the start hypotheses
            % match
            if isequal(hypotheses(j).measHistory(1:end-slideWindow+1,2), thisHyp)
                idx_remain(j) = true;
            end
        end
    else
        idx_remain(track_index{i}) = true;
    end
    %end
end

hypothesesOut = hypotheses(idx_remain);

%--------------------------------------------------------------------------

function [A0, A, track_index, measTrack_index, trackNull_index] =...
    getConstraints(hypotheses, historylength)

% Get unique track ids, and which hypotheses correspond to each track
[uqTrackIds, ~, track_index] = uniquewithcounts([hypotheses.trackID]);
numHypotheses = numel(hypotheses);
numTracks = numel(uqTrackIds);
% Get history length of each hypothesis
hypothesisHistLen = arrayfun(@(x)size(x.measHistory, 1), hypotheses);

% construct binary indicator matrix for constraint (1): each track should
% only be used once; this constraint is neccessary for the implementation of dual
% decomposition, since sliding window is used.
A0 = zeros(numTracks, numHypotheses);
% for each track
for t = 1:numTracks
    A0(t, track_index{t}) = 1;
end

% construct binary indicator matrix for constraint (2): each measurement in
% each scan should only be used once
A = cell(historylength, 1);

% For each scan in the history (going backwards in time)
measTrack_index = cell(historylength, 1);
trackNull_index = cell(historylength, numTracks);
for k = 1:historylength
    % get which measurements assigned for each hypothesis at this time scan
    % (if history is too short, assign null)
    idx = find(hypothesisHistLen>=k);
    meastemp = zeros(1, numHypotheses);
    meastemp(idx) = arrayfun(@(x) x.measHistory(end-k+1,2), hypotheses(idx));
    % get assigned measurements in this scan (not including null)
    measUnique = unique(meastemp(meastemp~=0));
    numMeas = numel(measUnique);
    % For each measurement, set constraint to be 1 for hypotheses which
    % assign it
    Atemp = false(numMeas, numHypotheses);
    for i = 1:numMeas
        Atemp(i, meastemp==measUnique(i)) = true;
    end
    A{k} = Atemp;
    
    % measTrack_index{k}{j, i} = hypothesis indices which assign
    %   measurement j to track i in scan k
    % trackNull_index{k, i} = hypothesis indices which assign the null
    %   hypothesis to track i in scan k
    measTrack_index{k} = cell(numMeas, numTracks);
    isnull = (meastemp==0);
    for i = 1:numTracks
        for j = 1:numMeas
            measTrack_index{k}{j, i} = track_index{i}(A{k}(j, track_index{i}));
        end
        trackNull_index{k, i} = track_index{i}(isnull(track_index{i}));
    end
end

%--------------------------------------------------------------------------

function [cost, idxCost, nullCost, idxNullCost] = get2dCostMatrix(...
    c_hat, measTrack_index, trackNull_index)

% Compute track-measurement cost for single scan problem from hypothesis
% cost, including null assignment costs

% construct track to measurement assignment matrix at scan k
[numMeas, numTracks] = size(measTrack_index);
cost = inf(numTracks, numMeas);
nullCost = inf(numTracks, 1);
% store index of the single target hypothesis with the minimum cost for
% each track and measurement
idxCost = zeros(numTracks, numMeas);
idxNullCost = zeros(numTracks, 1);

for i = 1:numTracks
    % find minimum cost and index of null hypotheses
    if ~isempty(trackNull_index{i})
        [nullCost(i), idxmin] = min(c_hat(trackNull_index{i}));
        idxNullCost(i) = trackNull_index{i}(idxmin);
    end
    % find single target hypotheses in track i that use this
    % measurement if found, find the single target hypothesis with
    % the minimum cost, and record its index
    for j = 1:numMeas
        if ~isempty(measTrack_index{j,i})
            [cost(i, j), idxmin] = min(c_hat(measTrack_index{j,i}));
            idxCost(i,j) = measTrack_index{j,i}(idxmin);
        end
    end
end

function assignments = assignmentoptimalNull(cost, nullCost)

% Get optimal assignments when we have null assignments

% Create cost matrix for nulls with null costs on diagonal and inf
% elsewhere (so we can have any number of null assignments)
[numTracks, numMeas] = size(cost);
nc = inf(numTracks, numTracks);
nc(1:(numTracks+1):end) = nullCost;
costInput = [cost nc];
costInput = costInput - min(costInput(:));
% Assume assignmentoptimal can handle non-square matrices
assignments = assignmentoptimal(costInput);
% Set null assignments to zero
assignments(assignments>numMeas) = 0;

%--------------------------------------------------------------------------

function [uprimalhat, primalCosthat] = getPrimalSolution(...
    u_hat_mean, Amatrix, hypothesisCosts, options)

% Get a primal (feasible but not necessarily optimal) solution from the
% dual solution:
% decompose problem into two parts - one where the dual subproblem
% solutions agree and the remaining part to be solved using intlinprog

% find partial primal solution without conflicts
idx_selectedHyps = u_hat_mean==1;

% % PRH: Why are we deselecting hypotheses with history length == slideWindow?
% idx_selectedHyps(hypothesisHistLen<=slideWindow) = false;

idx_unselectedHyps = ~idx_selectedHyps;
% Tracks and measurements not used by the partial solution (ordered by
% tracks first, then measurements for each scan)
idx_uncertainTracksMeas = sum(Amatrix(:,idx_selectedHyps),2)==0;

% If a track or measurement used by the partial solution, remove it from
% the problem to be solved by integer linear programming
for i = 1:numel(idx_uncertainTracksMeas)
    if ~idx_uncertainTracksMeas(i)
        idx_unselectedHyps(Amatrix(i,:)==1) = false;
    end
end

% Solve remaining problem by implementing branch and bound algorithm to
% find a feasible solution
A_uncertain = Amatrix(idx_uncertainTracksMeas, idx_unselectedHyps);
c_uncertain = hypothesisCosts(idx_unselectedHyps);
len_c_uncertain = length(c_uncertain);
Aeq = sparse(A_uncertain);
beq = ones(size(A_uncertain,1),1);
lower_bound = zeros(len_c_uncertain,1);
upper_bound = ones(len_c_uncertain,1);
[uprimal_uncertain, ~, exitflag] = intlinprog(c_uncertain, 1:len_c_uncertain, [], [],...
    Aeq, beq, lower_bound, upper_bound, [], options); %#ok<ASGLU>
uprimal_uncertain = round(uprimal_uncertain);

% Get solution to full problem by combining the partial and linear
% programming solutions
uprimalhat = u_hat_mean==1;
uprimalhat(idx_unselectedHyps) = logical(uprimal_uncertain);

% obtain primal cost
primalCosthat = hypothesisCosts'*uprimalhat;

% check that the obtained result is indeed feasible
if any(Amatrix*uprimalhat ~= 1)
    warning('getPrimalSolution - non-feasible solution returned')
end
