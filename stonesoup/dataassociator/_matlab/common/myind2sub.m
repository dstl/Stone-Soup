function sub = myind2sub(sz, ind)

% sub = myind2sub(sz, ind)
%
% Convert linear indices to multiple indices
% sub(i,:) = multiple index version of ind(i)

nd = numel(sz);
nind = numel(ind);
sub = zeros(nind, nd);
for i=1:nd
    sub(:,i) = mod(ind-1, sz(i))+1;
    ind = floor((ind-1)/sz(i))+1;
end
