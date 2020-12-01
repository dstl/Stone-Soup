function [uq, cnt, idx] = uniquewithcounts(v)

% [uq, cnt, idx] = uniquewithcounts(v)
%
% Get the unique values of a vector with the number of counts for each
%
% In:  v:  vector of numbers
% Out: uq:   1*n vector equal to unique(v)
%      cnts: 1*n vector where cnt(i) = sum(v==uq(i))
%      idx:  1*n cellarray where idx{i} = find(v==uq(i))

[vsort, p] = sort(v(:));
if isnumeric(v)
    lastidx = [0; find(diff([vsort;inf]))];
else
    [~,~,j] = unique(vsort);
    lastidx = [0; find(diff([j;inf]))];
end
uq = vsort(lastidx(2:end))';
cnt = diff(lastidx)';

if nargout>2
    idx = cell(1,numel(uq));
    for i=1:numel(uq)
        idx{i} = p(lastidx(i)+1:lastidx(i+1))';
    end
end