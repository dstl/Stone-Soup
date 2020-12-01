function Xc= get_comps(X,c)
if isempty(X)
    Xc= zeros(2,0);
else
    Xc= X(c,:);
end

