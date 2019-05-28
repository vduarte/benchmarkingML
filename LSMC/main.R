Spot = 36
σ = 0.2
n <- 100000
m <- 10
K <- 40
r <- 0.06
T <- 1
order <- 25
Δt <- T / m
zeros <- matrix(0, n, 1)


first_one_np <- function(x) {
    original <- x
    x <- x > 0
    n_columns <- dim(x)[2]
    batch_size <- dim(x)[1]
    x_not <- 1 - x
    sum_x <- pmin(t(apply(x_not, 1, cumprod)), 1.)
    ones <- matrix(1, batch_size, 1)
    lag <- sum_x[, 1:(n_columns - 1)]
    lag <- cbind(ones, lag)
    return (original * (lag * x))
}


chebyshev_basis <- function(x, k) {
    B = matrix(1, n, k)
    B[,2] = x
    for (count in seq(3, k)) {
        B[, count] = 2 * x * B[, count - 1] - B[, count - 2]
    }
    return(B)
}    


ridge_regression <- function(X, Y, lam){
    I = diag(nrow=order, ncol=order)
    beta = solve(t(X) %*% X + lam * I, t(X) %*%  Y)
    return(X %*% beta)
}


scale <- function(x){
        xmin = min(x)
        xmax = max(x)
        a = 2 / (xmax - xmin)
        b = -0.5 * a * (xmin + xmax)
        return(a * x + b)
}

advance <- function(S){
    set.seed(42)
    dB <- sqrt(Δt) * rnorm(n, 0, 1)
    out <- S + r * S * Δt + σ * S * dB
    return(out)
}

main <- function(){
    S <- matrix(0, n, m + 1)
    CFL <- matrix(0, n, m + 1)
    S[, 1] <- Spot

    for (count in seq(2, m + 1)) {
        S[, count] = advance(S[, count - 1])
    }    
    for (count in seq(1, m + 1)) {
        CFL[, count] =  pmax(0., K - S[, count])
    }    

    discount = exp(-r * Δt)

    value <- matrix(0, n, m + 1)
    value[, m + 1] <- CFL[, m + 1] * discount
    CV = matrix(0, n, m + 1)

    for (t in seq(m, 2)) {    
        t_next = t + 1
        XX = chebyshev_basis(scale(S[,t]), order)
        YY = value[,t_next]
        CV[, t] = ridge_regression(XX, YY, 100)
        value[, t] = discount * ifelse(CFL[, t] > CV[, t],
                                       CFL[,t],
                                       value[,t_next])
    }    

    POF <- matrix(0, n, m)
    for (count in seq(m)) {
        POF[, count] <- ifelse(CV[, count + 1] > CFL[, count + 1], matrix(0, n, 1), CFL[, count + 1])
    }

    FPOF = first_one_np(POF)
    dFPOF <- matrix(0, n, m)
    for (count in seq(1, m)) {
        dFPOF[, count] = t(FPOF[, count]) * exp(-r * (count - 1) * Δt)
    }
    PRICE = mean(rowSums(dFPOF))
    return(PRICE)    
}

repeats = 10
t0 <- Sys.time()
for (t in seq(repeats))
    main()
t1 <- Sys.time()
print((t1 - t0) / repeats * 4 * 1000)  # Multiply by four bc we need 4 greeks
