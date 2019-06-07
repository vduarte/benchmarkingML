function [ output ] = advance(S,delta_t,n,r,sigma)
    dB = sqrt(delta_t)*randn(1,n);
    output = S + r*S*delta_t + sigma*S.*dB;
end

