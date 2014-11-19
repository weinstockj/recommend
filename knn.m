function perf = knn(X, Xt, k, i, bigData)
    % for each user u and movie m
    nu = size(X, 1);
    nm = size(X, 2);

    %remove overall mean
    overallMean = sum(sum(X))/sum(sum(X ~= 0));
    % X = X - overallMean; 

    avgRu = sum(X, 2)./sum((X ~= 0), 2);
    avgRm = sum(X, 1)./sum((X ~= 0), 1);

    userEff = avgRu - overallMean;
    movieEff = avgRm - overallMean;
    movieEff(isnan(movieEff)) = 0;


    %user effect
    %userEffmat = repmat(userEff, 1, nm);
    X = bsxfun(@minus, X, userEff);

    %movie effect
    %movieEffmat = repmat(movieEff, nu, 1);
    X = bsxfun(@minus, X, movieEff);
    
    %Xnorm = X - userEffmat - movieEffmat - overallMean;
    Xnorm = X - overallMean;
    Xnorm(X == 0) = 0;
    
    avgRuNorm = sum((Xnorm ~= 0), 2)./sum((X ~= 0), 2);
    sim = [];
 if (bigData == 0)
     
     if (k == 5)
        sim = zeros(nm, nm);
        for mi = 1:nm
            for mj = 1:mi - 1
                % voters who have voted for both i and j
                commonVoters = find((X(:, mi) ~= 0) .* ((X(:, mj) ~= 0)));
                ri = Xnorm(commonVoters, mi);% - avgRuNorm(commonVoters);% - avgRm(mi) - overallMean;
                rj = Xnorm(commonVoters, mj);% - avgRuNorm(commonVoters);% - avgRm(mi) - overallMean;
                % movie similarity
                sim(mi, mj) = abs(sum(ri .* rj) / sqrt(sum(ri .^2) * (sum(rj .^ 2))));
                sim(mj, mi) = sim(mi, mj);
            end
        end
        %sim = max(sim, sim');

        sim(isnan(sim)) = 0;
        %sim = max(sim, sim');
        save(['msim' int2str(i) '.mat'], 'sim');
     else
         load(['msim' int2str(i) '.mat'], 'sim');
     end
 end
     
    %test
    Rout = zeros(length(Xt), 1);
    for i = 1: length(Xt(:,1))
        i
        Rout(i) = knnPredict(Xnorm, X, k, Xt(i, 1), Xt(i, 2), bigData, sim);
    end

    Rout (isnan(Rout)) = avgRuNorm(Xt(isnan(Rout), 1));
    Rout = Rout + userEff(Xt(:,1)) + movieEff(Xt(:,2))' + overallMean;
    perf = sqrt(sum((Rout - Xt(:, 3)) .^ 2) / length(Xt))

end
