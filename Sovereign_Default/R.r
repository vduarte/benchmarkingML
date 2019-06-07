logy_grid <- as.matrix(read.csv('logy_grid.txt', header = FALSE, sep = ' '))
Py <- as.matrix(read.csv('P.txt', header = FALSE, sep = ' '))

dmain <- function(logy_grid, Py, nB, repeats){
    beta <- 0.953
    r <- 0.017
    theta <- 0.282
    theta_comp <- 1 - theta

    ny <- nrow(logy_grid)

    Bgrid <- seq(-0.45,0.45, by = (0.45-(-0.45))/(nB-1) )
    ygrid <- exp(logy_grid)

    ymean <- mean(ygrid)
    def_y <- pmin(ygrid, 0.969 * ymean)
    d_gamma <- 2.0
    d_one_minus_gamma <- 1-d_gamma
    u_def_y <- def_y^d_one_minus_gamma / d_one_minus_gamma

    Vd <- matrix(data = 0, nrow = ny, ncol = 1)
    Vc <- matrix(data = 0, nrow = ny, ncol = nB)
    V <- matrix(data = 0, nrow = ny, ncol = nB)
    Q <- matrix(data = 0.95, nrow = ny, ncol = nB)

    myY <- matrix(data = ygrid, nrow = ny, ncol = 1)
    myBnext <- matrix(data = Bgrid, nrow = 1, ncol = nB)
    zero_ind <- nB %/% 2 + 1
    
    fun_multiplication <- function(x) { x * myBnext }
    fun_summation <- function(x) { x + myY }
    fun_comparison <- function(x) { Vd > x }
    fun_power <- function(x) { x^d_one_minus_gamma / d_one_minus_gamma }
    fun_pmax <- function(x) { pmax(x, Vd) }
    
    t0 <- Sys.time()
    for (i in 1:repeats) {
      betaEV <- beta * Py %*% V
      EVd <- Py %*% Vd
      EVc_zero_ind <- Py %*% Vc[,zero_ind]
      Vd_target <- u_def_y + beta * (theta * EVc_zero_ind + theta_comp * EVd)
      
      sliceTemplate <- -Q
      sliceTemplate <- t(apply(sliceTemplate, 1, fun_multiplication))
      sliceTemplate <- apply(sliceTemplate, 2, fun_summation)
      
      Vc_target <- matrix(nrow = ny, ncol = nB)
      for(j in 1:nB) {
        oneSlice <- sliceTemplate + Bgrid[j]
        oneSlice[oneSlice < 1e-14] <- 1e-14
        oneSlice <- oneSlice^d_one_minus_gamma / d_one_minus_gamma + betaEV
        Vc_target[,j] <- apply(oneSlice, 1, max)
      }
      
      default_states <- apply(Vc, 2, fun_comparison)
      default_prob <- Py %*% default_states
      
      Q <- (1 - default_prob) / (1 + r)
      V <- apply(Vc, 2, fun_pmax)
      Vc <- Vc_target
      Vd <- Vd_target
    }
    t1 <- Sys.time()
    out <- (t1 - t0) / repeats
    
    return(list(V = V, sec = as.double(out, units = "secs")))
}

res <- dmain(logy_grid, Py, 151, 100)
print(res$sec * 1000)
