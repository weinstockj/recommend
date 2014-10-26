function rating = knnPredict (X, k, msim, u, m)
    x = msim( : , m);
    x(X(u,:) == 0) = -1;
    nRatedByU = full(sum(X(u,:) > 0));
    [val, ind] = sort(x, 'descend');
    knns = ind(1:max(nRatedByU, k));
    rating = sum(msim(knns,m)' .* X(u, knns)) / sum(msim(knns,m));   
end
