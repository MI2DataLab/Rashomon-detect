derivative_fraction_sign_difference <- function(profile1, profile2, xargs){
  profile1der <- doremi::calculate.gold(time = xargs,
                                signal = profile1,
                                embedding = 4,
                                n = 1)
  profile2der <- doremi::calculate.gold(time = xargs,
                                signal = profile2,
                                embedding = 4,
                                n = 1)
  spline1 <- stats::spline(profile1der$dsignal[,2] ~ profile1der$dtime)
  spline1$y[abs(spline1$y) < 0.01 * max(abs(spline1$y))] <- 0
  spline2 <- stats::spline(profile2der$dsignal[,2] ~ profile2der$dtime)
  spline2$y[abs(spline2$y) < 0.01 * max(abs(spline2$y))] <- 0
  mean(sign(spline1$y) != sign(spline2$y))
}

derivative_fraction_sign_difference_simplified <- function(profile1, profile2, xargs){
  derivative1 <- fda.usc::fdata.deriv(fda.usc::fdata(profile1, argvals = xargs))$data[1,]
  derivative2 <- fda.usc::fdata.deriv(fda.usc::fdata(profile2, argvals = xargs))$data[1,]
  mean(sign(derivative1) != sign(derivative2))
}

euclidean_distance <- function(profile1, profile2, xargs){
  as.numeric(fda.usc::metric.lp(
    fda.usc::fdata(profile1, argvals = xargs),
    fda.usc::fdata(profile2, argvals = xargs)
  ))
}

derivative_euclidean_distance <- function(profile1, profile2, xargs){
  as.numeric(fda.usc::semimetric.deriv(
    fda.usc::fdata(profile1, argvals = xargs),
    fda.usc::fdata(profile2, argvals = xargs)
  ))
}

vector_distance <- function(profile1, profile2, xargs){
  stats::dist(rbind(profile1, profile2))[1]
}
