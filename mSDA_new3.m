function [allhx, Ws] = mSDA_new3(xx, firstDomainNum, noise,layers)

% xx : dxn input
% noise: corruption level
% layers: number of layers to stack

% allhx: (layers*d)xn stacked hidden representations

lambda = 1e-05;
disp('stacking hidden layers...')
prevhx = xx;
clear xx;
allhx = [];
Ws = {};
for layer = 1:layers
%     	disp(['layer:',num2str(layer)])
% 	tic
	[newhx, W] = mDA_new3(prevhx,firstDomainNum,noise,lambda);
	Ws{layer} = W;
%     newhx = newhx - min(min(newhx));
%     idx = newhx < 0;
%     newhx(idx) = 0;
%     newhx = normc(newhx);
% 	toc
	allhx = [allhx; newhx];
%     allhx = [allhx; layer*normc(newhx)];
%     allhx = [newhx];
	prevhx = newhx;
end
