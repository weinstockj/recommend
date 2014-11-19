function rating = knnPredict (X, Xorig, k, u, m, bigData, sim)
    if (bigData == 1)
        msim = similCalc(X, Xorig, m);
    else
        msim = sim(:,m);
    end        
    msim(X(u,:) == 0) = -1;
    nRatedByU = full(sum(X(u,:) > 0));
    [val, ind] = sort(msim, 'descend');
    knns = ind(1:max(nRatedByU, k));
    rating = sum(msim(knns)' .* X(u, knns)) / sum(msim(knns));   
end
