# ------------------------------------------------------------
# 1.3 Kalman filter
# ------------------------------------------------------------

myKalmanFilter <- function(
    y,
    theta,
    R,
    x_prior = 0,
    P_prior = 10
) {
  y <- as.numeric(y)
  
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]
  
  Q <- sigma1^2
  N <- length(y)
  
  x_pred <- numeric(N)
  P_pred <- numeric(N)
  x_filt <- numeric(N)
  P_filt <- numeric(N)
  innovation <- numeric(N)
  innovation_var <- numeric(N)
  
  x_pred_next <- numeric(N)
  P_pred_next <- numeric(N)
  
  for (t in seq_len(N)) {
    
    # Prediction of X_t before observing Y_t
    if (t == 1) {
      x_pred[t] <- x_prior
      P_pred[t] <- P_prior
    } else {
      x_pred[t] <- a * x_filt[t - 1] + b
      P_pred[t] <- a^2 * P_filt[t - 1] + Q
    }
    
    # Update with observation Y_t
    innovation[t] <- y[t] - x_pred[t]
    innovation_var[t] <- P_pred[t] + R
    
    K_t <- P_pred[t] / innovation_var[t]
    
    x_filt[t] <- x_pred[t] + K_t * innovation[t]
    P_filt[t] <- (1 - K_t) * P_pred[t]
    
    # Prediction of X_{t+1|t}
    x_pred_next[t] <- a * x_filt[t] + b
    P_pred_next[t] <- a^2 * P_filt[t] + Q
  }
  
  return(list(
    x_pred = x_pred,
    P_pred = P_pred,
    x_filt = x_filt,
    P_filt = P_filt,
    innovation = innovation,
    innovation_var = innovation_var,
    x_pred_next = x_pred_next,
    P_pred_next = P_pred_next
  ))
}


# ------------------------------------------------------------
# 1.4 Negative log-likelihood function
# ------------------------------------------------------------

myLogLikFun <- function(theta, y, R, x_prior = 0, P_prior = 10) {
  
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]
  
  # sigma1 must be positive
  if (sigma1 <= 0) {
    return(Inf)
  }
  
  kf_result <- myKalmanFilter(
    y = y,
    theta = c(a, b, sigma1),
    R = R,
    x_prior = x_prior,
    P_prior = P_prior
  )
  
  err <- kf_result$innovation
  S <- kf_result$innovation_var
  
  # Variances must be positive
  if (any(S <= 0)) {
    return(Inf)
  }
  
  # Log-likelihood from Gaussian innovations
  logL <- sum(-0.5 * (log(2 * pi * S) + (err^2 / S)))
  
  # Return negative log-likelihood for minimization
  return(-logL)
}


# ------------------------------------------------------------
# Data simulation function
# ------------------------------------------------------------

simulate_system <- function(a, b, sigma1, sigma2, X0, n) {
  
  X_t <- numeric(n + 1)
  Y_t <- numeric(n + 1)
  
  X_t[1] <- X0
  Y_t[1] <- X_t[1] + rnorm(1, mean = 0, sd = sigma2)
  
  for (t in 2:(n + 1)) {
    X_t[t] <- a * X_t[t - 1] + b + rnorm(1, mean = 0, sd = sigma1)
    Y_t[t] <- X_t[t] + rnorm(1, mean = 0, sd = sigma2)
  }
  
  return(list(
    X_t = X_t,
    Y_t = Y_t
  ))
}


# ------------------------------------------------------------
# One estimation study
# ------------------------------------------------------------

run_estimation_study <- function(
    a_true,
    b_true,
    sigma1_true,
    sigma2 = 1,
    X0 = 5,
    n = 100,
    n_sims = 100
) {
  
  estimates <- matrix(NA, nrow = n_sims, ncol = 3)
  colnames(estimates) <- c("a", "b", "sigma1")
  
  R <- sigma2^2
  
  for (i in 1:n_sims) {
    
    sim <- simulate_system(
      a = a_true,
      b = b_true,
      sigma1 = sigma1_true,
      sigma2 = sigma2,
      X0 = X0,
      n = n
    )
    
    Y_t <- sim$Y_t
    
    fit <- optim(
      par = c(0.5, 0.5, 1),
      fn = myLogLikFun,
      y = Y_t,
      R = R,
      x_prior = X0,
      P_prior = 10,
      method = "L-BFGS-B",
      lower = c(-Inf, -Inf, 0.001),
      upper = c(Inf, Inf, Inf)
    )
    
    if (fit$convergence == 0) {
      estimates[i, ] <- fit$par
    }
  }
  
  estimates <- estimates[complete.cases(estimates), ]
  
  return(estimates)
}


# ------------------------------------------------------------
# Run the three required cases
# ------------------------------------------------------------

set.seed(1)

cases <- list(
  list(
    params = c(0.9, 1, 1),
    label = "Base case: a = 0.9, b = 1, sigma1 = 1"
  ),
  list(
    params = c(0.9, 5, 1),
    label = "High bias: a = 0.9, b = 5, sigma1 = 1"
  ),
  list(
    params = c(0.9, 1, 5),
    label = "High process noise: a = 0.9, b = 1, sigma1 = 5"
  )
)

results <- list()

for (i in seq_along(cases)) {
  
  p <- cases[[i]]$params
  
  results[[i]] <- run_estimation_study(
    a_true = p[1],
    b_true = p[2],
    sigma1_true = p[3],
    sigma2 = 1,
    X0 = 5,
    n = 100,
    n_sims = 100
  )
}


# ------------------------------------------------------------
# Boxplots of estimates with improved y-axes
# ------------------------------------------------------------

par(mfrow = c(3, 1), mar = c(4, 4, 3, 1))

for (i in seq_along(cases)) {
  
  p <- cases[[i]]$params
  estimates <- results[[i]]
  
  # Choose y-axis limits based on estimates and true values
  y_min <- floor(min(c(estimates, p), na.rm = TRUE))
  y_max <- ceiling(max(c(estimates, p), na.rm = TRUE))
  
  # Add space around the plot
  y_min <- y_min - 1
  y_max <- y_max + 1
  
  # Integer y-axis ticks
  y_ticks <- seq(y_min, y_max, by = 1)
  
  boxplot(
    estimates,
    names = c("a", "b", expression(sigma[1])),
    main = cases[[i]]$label,
    ylab = "Estimated value",
    col = "lightgray",
    ylim = c(y_min, y_max),
    yaxt = "n"
  )
  
  axis(
    side = 2,
    at = y_ticks,
    labels = y_ticks,
    las = 1,
    cex.axis = 0.8
  )
  
  grid(
    nx = NA,
    ny = NULL,
    col = "gray85",
    lty = "dotted"
  )
  
  points(
    x = 1:3,
    y = p,
    col = "red",
    pch = 19,
    cex = 1.2
  )
  
  legend(
    "topright",
    legend = "True value",
    col = "red",
    pch = 19,
    bty = "n"
  )
}

for (i in seq_along(cases)) {
  
  cat("\n", cases[[i]]$label, "\n")
  
  cat("True values:\n")
  print(c(
    a = cases[[i]]$params[1],
    b = cases[[i]]$params[2],
    sigma1 = cases[[i]]$params[3]
  ))
  
  cat("Mean estimates:\n")
  print(colMeans(results[[i]]))
  
  cat("Standard deviations:\n")
  print(apply(results[[i]], 2, sd))
}