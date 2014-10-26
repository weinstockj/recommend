[X, U, M, R] = readMovieLense('ml-100k/u1.base');

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
userEffmat = repmat(userEff, 1, nm);

%movie effect
movieEffmat = repmat(movieEff, nu, 1);

Xnorm = X - userEffmat - movieEffmat - overallMean;
Xnorm(X == 0) = 0;

avgRuNorm = sum(Xnorm, 2)./sum((Xnorm ~= 0), 2);
avgRmNorm = sum(Xnorm, 1)./sum((Xnorm ~= 0), 1);

msim = zeros (nm, nm);

for mi = 1:nm
    for mj = 1:mi - 1
        % voters who have voted for both i and j
        commonVoters = find((X(:, mi) ~= 0) .* ((X(:, mj) ~= 0)));
        ri = Xnorm(commonVoters, mi);% - avgRuNorm(commonVoters);% - avgRm(mi) - overallMean;
        rj = Xnorm(commonVoters, mj);% - avgRuNorm(commonVoters);% - avgRm(mi) - overallMean;
        % movie similarity
        msim(mi, mj) = sum(ri .* rj) / sqrt(sum(ri .^2) * (sum(rj .^ 2)));
    end
end
msim = max (msim, msim');

%test

Xt = dlmread('ml-100k/u1.test', '\t');
%Xt = dlmread('ml-100k/u1.base', '\t');
Rout = zeros(length(Xt), 1);
for i = 1: length(Xt(:,1))
    Rout(i) = knnPredict(Xnorm, 5, msim, Xt(i, 1), Xt(i, 2));
end

Rout (isnan(Rout)) = avgRuNorm(Xt(isnan(Rout), 1));
Rout = Rout + userEff(Xt(:,1)) + movieEff(Xt(:,2))' + overallMean;
perf = sqrt(sum((Rout - Xt(:, 3)) .^ 2) / length(Xt));

