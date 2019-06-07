function [output] = first_one(x)
    original = x;
    for i=1:length(x(:,1))
        for j=1:length(x(1,:))
            if x(i,j)>0
                x(i,j)=1;
            else
                x(i,j)=0;
            end
        end
    end
    x_not=not(x);
    n_columns = length(x(1,:));
    batch_size = length(x(:,1));
    sum_x=min(cumprod(x_not,2),1.);
    v_ones = ones(batch_size,1);
    lag = sum_x(:, 1:(n_columns - 1));
    lag = [v_ones lag];
    output= original.*(lag.* x);  
end

