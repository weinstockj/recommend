perfs = zeros(8, 1);


[X] = readMovieLense('ml-100k/u1.base');
Xt = dlmread('ml-100k/u1.test', '\t');
i = 1;
f = 1;
for k = [5 10 20 30 40 50 60 70]
    perfs(i) = perfs(i) + knn(X, Xt, k, f, 0);
    i = i + 1;
end

[X] = readMovieLense('ml-100k/u5.base');
Xt = dlmread('ml-100k/u5.test', '\t');
f = f + 1;
i = 1;

for k = [5 10 20 30 40 50 60 70]
    perfs(i) = perfs(i) + knn(X, Xt, k, f, 0);
    i = i + 1
end


[X] = readMovieLense('ml-100k/u4.base');
Xt = dlmread('ml-100k/u4.test', '\t');
i = 1;
f = f + 1;
for k = [5 10 20 30 40 50 60 70]
    perfs(i) = perfs(i) + knn(X, Xt, k, f, 0);
    i = i + 1;
end


[X] = readMovieLense('ml-100k/u3.base');
Xt = dlmread('ml-100k/u3.test', '\t');
i = 1;
f = f + 1;
for k = [5 10 20 30 40 50 60 70]
    perfs(i) = perfs(i) + knn(X, Xt, k, f, 0);
    i = i + 1;
end

[X] = readMovieLense('ml-100k/u2.base');
Xt = dlmread('ml-100k/u2.test', '\t');
i = 1;
f = f + 1;
for k = [5 10 20 30 40 50 60 70]
    perfs(i) = perfs(i) + knn(X, Xt, k, f, 0);
    i = i + 1;
end



perfs = perfs ./ 5
