% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,000 features
% use liblinear  other than the libSVM

clear all;
close all;
addpath('./lowDimen')
addpath('./liblinear-1.93/');
domains=cell(4,1);
domains{1}='books';
domains{2}='dvd';
domains{3}='electronics';
domains{4}='kitchen';

% folds = 5;
% two hyper-parameters to be cross-validated
% number of mSDA layers to be stacked
layers = 5;
% corruption level
noises = [0.5,0.6,0.7,0.8,0.9];

% read in the raw input
load('amazon.mat');
dimen = 5000;% oringinal is 5000
xx = xx(1:dimen, :);

% finalize training and testing
bestNoise = 0.7;
% [allhx] = mSDA(double(xx), bestNoise, layers);
% [temp, noiseIdx] = max(ACCs);
testCase = 0;
acc_total = 0;
for j = 1:size(domains,1)
    % 	[allhx] = mSDA(double(xx>0), bestNoise, layers);
    % 	xr=[xx(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
    % 	xr=xr';
    % 	bestC=Cs(noiseIdx(j),j);
    % 	disp(['final training on domain ', source, ' ...'])
    % 	model = svmtrain_libSVM(yr,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
    %   clear xr yr;
    for i = 1:size(domains,1)
        if i == j
            continue;
        end
%         if j ~= 3 || i ~= 1 % just the current
%             continue;
%         end
        source = domains{j};
        %         bestNoise = noises(noiseIdx(j));
%         bestNoise = 0.7;
        disp(['learn representation with corruption level ' num2str(bestNoise), ' ...']);
        
        % unsupervised feature learning using samples in source domain and target domain        
        cmp_idx = [offset(j)+1:offset(j+1) offset(i)+1:offset(i+1)];
        [allhx] = mSDA_new3(double(xx(:,cmp_idx)), offset(j+1)-offset(j)+1, bestNoise, layers);

        train_idx = offset(j)+1:offset(j+1);
%         xr = [xx(:,train_idx);allhx(:, 1:offset(j+1)-offset(j))];
        xr = [allhx(:, 1:offset(j+1)-offset(j))];
        xr = xr';
        yr = yy(train_idx);

        
        
        bestC = 0.01;
        disp(['final training on domain ', source, ' ...'])
%         model = svmtrain_libSVM(yr,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
        model = train_liblinear(yr,sparse(xr),['-q -c ',num2str(bestC)]);
%         clear xr yr;
        
        target = domains{i};
        test_idx = offset(i)+1:offset(i+1);
        disp(['final testing on domain ', target, ' ...'])
%         xe = [xx(:, offset(i)+2001:offset(i+1))];
%         xe = [xx(:,test_idx);allhx(:, offset(j+1)-offset(j)+1:end)];
        xe = [allhx(:, offset(j+1)-offset(j)+1:end)];
        xe = xe';
        ye = yy(test_idx);
        [label,accuracy,prob] = predict_liblinear(ye,sparse(xe),model);
        acc_total = acc_total + accuracy(1);
        testCase = testCase + 1;
%         clear xe ye;
        fprintf('\n');
    end
end
acc_mean = acc_total / testCase

%% un-supervised results:, dimen = 5000;, combine feature
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain books ...
% final testing on domain dvd ...
% Accuracy = 82.5815% (4613/5586)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain books ...
% final testing on domain electronics ...
% Accuracy = 80.2109% (6161/7681)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain books ...
% final testing on domain kitchen ...
% Accuracy = 82.769% (6576/7945)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain dvd ...
% final testing on domain books ...
% Accuracy = 80.9435% (5233/6465)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain dvd ...
% final testing on domain electronics ...
% Accuracy = 80.8619% (6211/7681)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain dvd ...
% final testing on domain kitchen ...
% Accuracy = 82.8194% (6580/7945)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain electronics ...
% final testing on domain books ...
% Accuracy = 74.9575% (4846/6465)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain electronics ...
% final testing on domain dvd ...
% Accuracy = 76.4769% (4272/5586)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain electronics ...
% final testing on domain kitchen ...
% Accuracy = 87.8918% (6983/7945)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain kitchen ...
% final testing on domain books ...
% Accuracy = 74.7718% (4834/6465)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain kitchen ...
% final testing on domain dvd ...
% Accuracy = 78.5714% (4389/5586)
% 
% learn representation with corruption level 0.7 ...
% stacking hidden layers...
% final training on domain kitchen ...
% final testing on domain electronics ...
% Accuracy = 86.1476% (6617/7681)
% 
% 
% acc_mean =
% 
%    80.7503

