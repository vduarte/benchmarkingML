function [output] = Scale(x)
        xmin = min(x);
        xmax = max(x);
        a = 2 / (xmax - xmin);
        b = -0.5 * a * (xmin + xmax);
        output=a*x + b;
end

