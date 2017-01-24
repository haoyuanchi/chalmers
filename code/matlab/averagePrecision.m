function [precision_at_k, recall_at_k, mAP] = averagePrecision(trn_label, trn_binary, tst_label, tst_binary, top_k, mode)

num_test = size(tst_binary,2);
precision = zeros(top_k, num_test);
recall = zeros(top_k, num_test);
averagePrecision = zeros(num_test, 1);

Ns = 1:1:top_k;

for i = 1 : num_test
    query_label = tst_label(i);
    fprintf('query %d\n',i);
    query_binary = tst_binary(:,i);
    if mode==1
    tic
    similarity = pdist2(trn_binary',query_binary','hamming');
    toc
    fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
    elseif mode ==2
    tic
    similarity = pdist2(trn_binary',query_binary','euclidean');
    toc
    fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
    end

    [x2, y2]=sort(similarity);

    % total train num that have the same label with test 
    % total_good_pairs = size(find((trn_label) == query_label), 1);
    total_good_pairs = top_k;
    %exp. # of good pairs that have exactly the same code
    retrieved_good_pairs = 0;
    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = top_k;

    buffer_yes = zeros(top_k, 1);  

    for j = 1:top_k
        retrieval_label = trn_label(y2(j));

        if (query_label==retrieval_label)            
            buffer_yes(j,1) = 1;
            retrieved_good_pairs = retrieved_good_pairs + 1;                
        end
    end

    % compute precision, every image precision represented by the column
    precision(:, i) = cumsum(buffer_yes) ./ Ns';
    recall(:, i) = cumsum(buffer_yes) ./ total_good_pairs;
    
    % compute the AP

    if (sum(buffer_yes) == 0)
        averagePrecision(i) = 0;
    else
        averagePrecision(i) = sum(precision(:, i).*buffer_yes) / sum(buffer_yes);
    end
end

precision_at_k = mean(precision, 2);
recall_at_k = mean(recall, 2);

mAP = mean(averagePrecision);

