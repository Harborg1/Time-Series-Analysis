myKalmanFilter <- function(
    y,              # Vector of observations y_t
    theta,          # Model parameters: theta = c(a, b, sigma1)
    R,              # Measurement noise variance = sigma2^2
    x_prior = 0,    # Initial prior mean for X_0
    P_prior = 10    # Initial prior variance for X_0
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


# Simulate data from 1.2
set.seed(1)

a <- 0.9
b <- 1
sigma1 <- 1
sigma2 <- 1
X0 <- 5
n <- 100

time <- 0:n

X_t <- numeric(n + 1)
Y_t <- numeric(n + 1)

X_t[1] <- X0
Y_t[1] <- X_t[1] + rnorm(1, mean = 0, sd = sigma2)

for (t in 2:(n + 1)) {
  X_t[t] <- a * X_t[t - 1] + b + rnorm(1, mean = 0, sd = sigma1)
  Y_t[t] <- X_t[t] + rnorm(1, mean = 0, sd = sigma2)
}


# Run Kalman filter
theta <- c(a, b, sigma1)
R <- sigma2^2

kf <- myKalmanFilter(
  y = Y_t,
  theta = theta,
  R = R,
  x_prior = X0,
  P_prior = 10
)


# 95% confidence interval around X_{t+1|t}
ci_lower <- kf$x_pred_next - 1.96 * sqrt(kf$P_pred_next)
ci_upper <- kf$x_pred_next + 1.96 * sqrt(kf$P_pred_next)


# Align X_{t+1|t} with the time it predicts
# kf$x_pred_next[1] predicts X_1, so it is plotted at time 1
pred_time <- 1:n

pred_mean <- kf$x_pred_next[1:n]
pred_lower <- ci_lower[1:n]
pred_upper <- ci_upper[1:n]


# Choose y-axis limits
y_min <- floor(min(c(X_t, Y_t, pred_mean, pred_lower, pred_upper), na.rm = TRUE))
y_max <- ceiling(max(c(X_t, Y_t, pred_mean, pred_lower, pred_upper), na.rm = TRUE))

# Show every integer value on the y-axis
y_ticks <- seq(y_min, y_max, by = 1)


# Plot
plot(
  time,
  X_t,
  type = "n",
  xlab = "t",
  ylab = "Value",
  main = "Kalman filter: Prediction and State Tracking",
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

polygon(
  c(pred_time, rev(pred_time)),
  c(pred_lower, rev(pred_upper)),
  col = adjustcolor("skyblue", alpha.f = 0.3),
  border = NA
)

lines(time, X_t, col = "red", lwd = 2)
points(time, Y_t, col = "gray40", pch = 16, cex = 0.7)
lines(pred_time, pred_mean, col = "blue", lwd = 2, lty = 2)

legend(
  "topleft",
  legend = c(
    "True state X_t",
    "Observation Y_t",
    "Predicted state X_{t+1|t}",
    "95% CI around X_{t+1|t}"
  ),
  col = c("red", "gray40", "blue", "skyblue"),
  lty = c(1, NA, 2, NA),
  pch = c(NA, 16, NA, 15),
  pt.cex = c(NA, 0.5, NA, 1.2),
  lwd = c(1.5, NA, 1.5, NA),
  bty = "n",
  cex = 0.65,
  y.intersp = 0.8,
  x.intersp = 0.6,
  seg.len = 1.2
)


# Optional checks
head(kf$x_pred_next)
head(kf$P_pred_next)
head(kf$innovation)
head(kf$innovation_var)


# Residuals between true state and predicted state X_{t+1|t}
# kf$x_pred_next[1:n] predicts X_t at times 1,...,n
# In the simulated series, those true values are X_t[2:(n + 1)]

pred_time <- 1:n
true_next <- X_t[2:(n + 1)]
pred_next <- kf$x_pred_next[1:n]

residuals_pred <- true_next - pred_next


# Choose y-axis limits for residual plot
res_y_min <- floor(min(residuals_pred, na.rm = TRUE))
res_y_max <- ceiling(max(residuals_pred, na.rm = TRUE))

# Show every integer value on the residual y-axis
res_y_ticks <- seq(res_y_min, res_y_max, by = 1)


plot(
  pred_time,
  residuals_pred,
  type = "l",
  col = "purple",
  lwd = 2,
  xlab = "t",
  ylab = "Residual",
  main = "Residuals: True state minus predicted state",
  ylim = c(res_y_min, res_y_max),
  yaxt = "n"
)

axis(
  side = 2,
  at = res_y_ticks,
  labels = res_y_ticks,
  las = 1,
  cex.axis = 0.8
)

abline(h = 0, col = "red", lty = 2)

points(
  pred_time,
  residuals_pred,
  pch = 16,
  cex = 0.6,
  col = "purple"
)


# Optional summary
mean(residuals_pred)
sd(residuals_pred)