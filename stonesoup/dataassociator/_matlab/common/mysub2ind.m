function ind = mysub2ind(sz, sub)

% ind = mysub2ind(sz, sub)
%
% Convert  multiple indices to linear indices
% ind(i) = linear index version of sub(i,:)

[nsub, nd] = size(sub);
ind = zeros(1, nsub);
for i=nd:-1:1
    ind = ind*sz(i) + (sub(:,i)-1)';
end
ind = ind + 1;
