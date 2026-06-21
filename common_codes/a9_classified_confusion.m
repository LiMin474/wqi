function a9_classified_confusion(X)
edges = [0 50 100 150 200 inf];
labels = [1,2, 3, 4, 5];
WQIclass = discretize(X, edges, labels);

[uniqueRows, ~, ic] = unique(WQIclass, 'rows');
counts = accumarray(ic, 1);
disp('Unique classification patterns and their counts:');
disp(table(uniqueRows, counts));

numMethods = size(WQIclass, 2);
counts = zeros(numel(labels), numMethods);
for m = 1:numMethods
    counts(:,m) = histcounts(X(:,m), edges);
end

% Create readable row labels if class numbers represent categories
rowNames = compose("Class_%d", labels);  % or use ["Good","Medium","Bad"] if known
colNames = compose("Method_%d", 1:numMethods);
% Convert to table for display
T = array2table(counts, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T);

disp('')