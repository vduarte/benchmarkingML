function [output] = compute_price(order, Spot, sigma, K, r)
    n = 100000;
    m = 10;
    T = 1;
    delta_t = T/m;
    S=zeros(m+1,n);
    S(1,:) = Spot*ones(1,n);
    for t=2:m+1
        S(t,:)=advance(S(t-1,:),delta_t,n,r,sigma);
    end
    discount = exp(-r * delta_t);
    CFL=max(0,K-S);
    value = zeros(m,n);
    value(end,:) = CFL(end,:)*discount;
    CV=zeros(m,n);
    for k=1:m-1
        t=m-k;
        t_next = t + 1;
        XX = chebyshev_basis(Scale(S(t+1,:)), order);
        YY = value(t_next,:);
        CV(t,:) = ridge_regression(XX, YY, 100);
        aux = where(CFL(t+1,:)>CV(t,:), CFL(t+1,:), value(t_next,:));
        value(t,:) = discount*aux;
    end
    POF = where(CV < CFL(2:end, :), CFL(2:end, :), 0 * CFL(2:end, :))';
    FPOF = first_one(POF);
    dFPOF = (FPOF.*exp(-r*(0:m-1)*delta_t));
    PRICE = mean(sum(dFPOF,2));
    output=PRICE;
end