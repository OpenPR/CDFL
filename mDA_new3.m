function [hx,W] = mDA_new3(xx,firstDomainNum,noise,lambda)

% xx : d1xn input features in first domain,
% yy : d2Xn input features in second domain,
% noise: corruption level
% lambda: regularization

% hx,hy: dxn hidden representation
% Wx,Wy: dx(d+1) mapping

[d, n] = size(xx);

%
n1 = firstDomainNum;
n2 = n - n1;

% adding bias
xxb = [xx; ones(1, n)];

clear xx;
% scatter matrix S
S = xxb*xxb';

% corruption vector
q = ones(d+1, 1)*(1-noise);
q(end) = 1;

% Q: (d+1)x(d+1)
Q = S.*(q*q');
Q(1:d+2:end) = q.*diag(S); % elements on the diagonal
% save('Q.mat','Q');

% P: dx(d+1)
P = S(1:end-1,:).*repmat(q', d, 1);
clear S;
% save('P.mat','P');

% G: 
G = sum(xxb(:,1:n1),2)/n1 - sum(xxb(:,n1+1:end),2)/n2;
G = G*G';
% G = G.*(q*q');% add noise
clear q;
% 
%get the positive part of G and Q
pos_G = G;
idx = pos_G < 0;
pos_G(idx) = 0;
pos_Q = Q;
idx = pos_Q < 0;
pos_Q(idx) = 0;
lambda_G = max(max(abs(pos_Q))) / max(max(abs(pos_G)));
clear pos_G pos_Q;

G = lambda_G*G;
% % reglarize
% % G = G + S;
% G = 20*n*G;
% save('G.mat','G');
clear S;


% final W = P*Q^-1, dx(d+1);
reg = lambda*eye(d+1);% set not to be zero
reg(end,end) = 0;
% W = P/(Q+reg);



% W = P / (Q - G - 2 * [[eye(size(Q)-1);zeros(1,size(Q,2)-1)] zeros(size(Q,1),1)] + reg);
W = P / (Q - 0.2*G + reg);
% W = P / (Q + reg);
% diff = Q-G;
% save('Q-G.mat','diff');
% clear diff;

% update W using gradient descent for several steps
% sum_2 = sum(xxb,2);
% for iter = 1:1050
%     for j = 1:size(W,1)
%         for k = 1:size(W,2)
% %             W(j,k)
% %             full(2/size(xxb,2)*sum_2(k)*(W(j,:)*sum_2/size(xxb,2) - 0.08))
%             W(j,k) = W(j,k) - 0.3 * 2/size(xxb,2)*sum_2(k)*(W(j,:)*sum_2/size(xxb,2) - 0.08);
% %             pause;
%         end
%     end
% end
% save('W2.mat','W');

clear P Q G reg;
hx = W * xxb;
clear xxb;
hx = tanh(hx);

