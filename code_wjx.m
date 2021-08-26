%% Read the data
r = csvread('return.csv',2,1,[2 1 241 40]);
% compute the yearly risk-free rate
rf = csvread('TNX.csv',1,4,[1 4 180 4]);
r0 = [];

for i = 1:15
    a = geo_mean(rf(12*i-11:12*i,:))/12*0.01+1;
    r0 = [r0,a];
end

%% model 1: mean-variance

m1_portfolio = [];% stores portfolios in columns
in_ret1 = []; % stores the in-sample return of portfolio
out_ret1 = []; % stores the out-of-sample return of portfolio
stdev1 = [];  % stores standard deviation of portfolio
equally_weighted2 = []; in_SR1=[];equally_weighted1=[];out_SR1=[];

for i = 1:15
    % 5 years historical return data
    ret1 = r(12*(i+4)-59:12*(i+4),:);
    % the following year return data
    ret2 = r(12*(i+5)-11:12*(i+5),:);
    
    % compute the geometric mean for in-sample and out-of-sample
    mu1 = geo_mean(ret1);
    mu2 = geo_mean(ret2);
    
    % equally weighted strategy
    equally_weighted2 = [equally_weighted2;mean(mu2)];
    equally_weighted1 = [equally_weighted1;mean(mu1)];
    
    % Compute Covariance matrix
    % compute each column's arithmetic mean
    avg = mean(ret1);  
    n = size(ret1,2);
    m = size(ret1,1);
    % Calculate the bottom triangular portion first.
    for i=1:n
        for j=1:i
            Sigma(i,j) = ((ret1(:,i) - avg(i))' * (ret1(:,j) - avg(j)))/m ;
        end
    end
    % Now add to the matrix by flipping it over.
    Sigma = Sigma + triu(Sigma',1);
    
    % Risk-adjusted Return model
    % We find the optimal investment for 2006-2021
    lambda = [0:2:20,30:10:100,150:50:300]; %  Risk-aversion coefficient could be changed
    % get the right sigma from new_sigma matrix for each year
    for k = 1:length(lambda)
        % cvx model
        cvx_begin quiet
        variable x(n);
        maximize(mu1*x - lambda(k)*x'*Sigma*x);
        ones(1,n)*x == 1;
        x >= 0;
        cvx_end
        
        m1_portfolio = [m1_portfolio x];
        in_ret1 = [in_ret1 mu1*x];
        out_ret1 = [out_ret1 mu2*x];
        stdev1 = [stdev1 sqrt(x'*Sigma*x)];
    end
    % plot
    %frontierarea2(in_ret1,stdev1,m1_portfolio,lambda,'lambda');
end

% compute the sharpe ratio of this model
for i = 1:23
    a = (in_ret1(15*i-14:15*i)-r0)./stdev1(15*i-14:15*i);
    b = (out_ret1(15*i-14:15*i)-r0)./stdev1(15*i-14:15*i);
    in_SR1 = [in_SR1;a];
    out_SR1 = [out_SR1;b];
    
end

%% m2: maximizing the sharpe ratio
m2_portfolio = [];
in_ret2 = []; 
out_ret2 =[];
stdev2 = [];  
in_SR2 = [];
out_SR2 = [];
K = [];
Z = [];

for i = 1:15
     % 5 years historical return data
    ret1 = r(12*(i+4)-59:12*(i+4),:);
    % the following year return data
    ret2 = r(12*(i+5)-11:12*(i+5),:);
    
    % compute the geometric mean
    mu1 = geo_mean(ret1);
    mu2 = geo_mean(ret2);
    
    % Compute Covariance matrix
    avg = mean(ret1);   % arithmetic mean of each column
    n = size(ret1,2);
    m = size(ret1,1);
    Sigma = zeros(n,n);
    % First compute the lower triangular part
    for k=1:n
        for j=1:k
            Sigma(k,j) = ((ret1(:,k) - avg(k))' * (ret1(:,j) - avg(j)))/m ;
        end
    end
    % Now flip the matrix and add
    Sigma = Sigma + triu(Sigma',1);
    
    % model
    cvx_begin
        variable z(n);
        variable kappa;
        minimize(z'*Sigma*z);
        z - kappa*ones(n,1) <= 0;
        mu1*z - r0(i)*kappa == 1;
        sum(z) - kappa == 0;
        z >= 0; 
        kappa >= 0;
    cvx_end 
    
    x = z/kappa; % optimal portfolio
    m2_portfolio = [m2_portfolio x];
    K = [K,kappa];
    Z = [Z,z];
    stdev2 = [stdev2 sqrt(x'*Sigma*x)];
    in_ret2 = [in_ret2 mu1*x];
    out_ret2 = [out_ret2 mu2*x];
    in_SR2 = (in_ret2(i) - r0(i))./stdev2;
    out_SR2 = (out_ret2(i) - r0(i))./stdev2;
end

%% m3: minimizing the CVaR
% set the value of beta
beta = [0.7:0.05:0.95,0.99];
n = 40;
m = 60;
prob = ones(1,60)*1/m; % the probabilities for these scenarios to occure
B = 1000;
L = 1050;
m3_portfolio = []; in_ret3 = []; out_ret3 = []; 
Cvar = []; stdev3=[];in_SR3=[];out_SR3=[];

for i=1:15
    % 5 years historical return data
    ret1 = r(12*(i+4)-59:12*(i+4),:);
    % the following year return data
    ret2 = r(12*(i+5)-11:12*(i+5),:);
    % compute the geometric mean
    mu1 = geo_mean(ret1);
    mu2 = geo_mean(ret2);
    
    % Compute Covariance matrix
    avg = mean(ret1);   % arithmetic mean of each column
    n = size(ret1,2);
    m = size(ret1,1);
    Sigma = zeros(n,n);
    % First compute the lower triangular part
    for k=1:n
        for j=1:k
            Sigma(k,j) = ((ret1(:,k) - avg(k))' * (ret1(:,j) - avg(j)))/m ;
        end
    end
    % Now flip the matrix and add
    Sigma = Sigma + triu(Sigma',1);
    
    for k = 1:length(beta)
        cvx_begin quiet
        variable x(n);  % proportion of investment in each asset
        variable z(m);
        variable t;
        minimize(t + prob*z/(1-beta(k)));
        z + ones(m,1)*t + B*ret1*x >= L;
        mu1*x >= r0(i);
        ones(1,n)*x == 1;
        x >= 0;
        z >= 0;
        cvx_end
        m3_portfolio = [m3_portfolio x];
        in_ret3 = [in_ret3, mu1*x];
        out_ret3 = [out_ret3, mu2*x];
        stdev3 = [stdev3 sqrt(x'*Sigma*x)];
        obj = t + prob*z/(1-beta(k));
        Cvar = [Cvar; obj];
    end
end
% compute the sharpe ratio for CVaR
for i = 1:7
    a = (in_ret3(15*i-14:15*i)-r0)./stdev3(15*i-14:15*i);
    b = (out_ret3(15*i-14:15*i)-r0)./stdev3(15*i-14:15*i);
    in_SR3 = [in_SR3;a];
    out_SR3 = [out_SR3;b];
end
