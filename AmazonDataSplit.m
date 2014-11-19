load('amazon.mat', 'X');
    [m, n] = size(X);
    X = X(randperm(m), :);

    
    % K fold cross validation
K = 3;
F = round(m / K);

    Xtest = X(F + 1: 2*F, :);
    Xval = X(2 * F + 1 : 3 * F, :);
    Xtrain = X(1:F, 1:2);
    ytrain = X(1:F, 3);
    
    uniqUsers = unique(Xtrain(:,1));
    indsTest = [];
    for i = 1:length(uniqUsers)
        i
        indsTest = union(indsTest, find(Xtest(:,1) == uniqUsers(i)));
    end
    indsVal = [];
    for i = 1:length(uniqUsers)
        i
        indsVal = union(indsVal, find(Xval(:,1) == uniqUsers(i)));
    end
        
    Xtest = Xtest(indsTest, 1:3);
    
    Xval = Xval(indsVal, 1:3);
    
    Xtrain = [Xtrain, ytrain];
    
    save(['Train.mat'], 'Xtrain');
    save(['Test.mat'],'Xtest');
    save(['Vaidation.mat'], 'Xval');
