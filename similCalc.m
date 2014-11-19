function sim = similCalc(X, Xorig, m)

    nm = size(X, 2);
    

        sim = zeros(nm, 1);
        for mj = 1:nm
        % voters who have voted for both i and j
        commonVoters = find((Xorig(:, m) ~= 0) .* ((Xorig(:, mj) ~= 0)));
        ri = X(commonVoters, m);% - avgRuNorm(commonVoters);% - avgRm(mi) - overallMean;
        rj = X(commonVoters, mj);% - avgRuNorm(commonVoters);% - avgRm(mi) - overallMean;
        % movie similarity
        sim(mj) = (sum(ri .* rj) / sqrt(sum(ri .^2) * (sum(rj .^ 2))));
        end


    sim(isnan(sim)) = 0;
end
