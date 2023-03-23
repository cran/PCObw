## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(PCObw)

## -----------------------------------------------------------------------------
# simple example with univariate data
data("gauss_1D_sample")
bw.L2PCO(gauss_1D_sample)




## -----------------------------------------------------------------------------
# simple example with epanechnikov kernel
data("gauss_1D_sample")
bw.L2PCO(gauss_1D_sample, K_name = "epanechnikov")

# simple example with biweight kernel
bw.L2PCO(gauss_1D_sample, K_name = "biweight")




## -----------------------------------------------------------------------------
# example when the tolerance is not reached
data("gauss_1D_sample")
bw.L2PCO(gauss_1D_sample, nh = 3)

bw.L2PCO(gauss_1D_sample, tol = 10^(-6))


## -----------------------------------------------------------------------------
# binning example
data("gauss_1D_sample")
bw.L2PCO(gauss_1D_sample, binning = TRUE)


## -----------------------------------------------------------------------------
# change the number of bins "nb"
data("gauss_1D_sample")
bw.L2PCO(gauss_1D_sample, binning = TRUE, nb = 130)

# or use "adapt_nb_bin = TRUE"
bw.L2PCO(gauss_1D_sample, binning = TRUE, adapt_nb_bin = TRUE)

# time comparison between exact and binned criterion with an huge sample
huge_sample <- rnorm(n = 10000, mean = 0, sd = 1)

ptm0 <- proc.time()
bw.L2PCO(huge_sample)
proc.time() - ptm0

ptm0 <- proc.time()
bw.L2PCO(huge_sample, binning = TRUE, adapt_nb_bin = TRUE)
proc.time() - ptm0


## -----------------------------------------------------------------------------

# example with 2D data
data("gauss_mD_sample")

# to return a full matrix
bw.L2PCO(gauss_mD_sample)

# to return a diagonal matrix
bw.L2PCO.diag(gauss_mD_sample)

## -----------------------------------------------------------------------------
data("gauss_mD_sample")

# increase the tolerance for faster results
bw.L2PCO.diag(gauss_mD_sample, tol = 10^(-3))

# increase "nh" for more accurate results
bw.L2PCO.diag(gauss_mD_sample, nh = 80)

# increase the tolerance for faster results
bw.L2PCO(gauss_mD_sample, tol = 10^(-3))

# increase "nh" for more accurate results
bw.L2PCO(gauss_mD_sample, nh = 80)


## -----------------------------------------------------------------------------
data("gauss_mD_sample")

# with a too small number of bins, the results are degenerated
bw.L2PCO.diag(gauss_mD_sample, binning = TRUE, nh = 80, nb_bin_vect = c(5, 10))

bw.L2PCO.diag(gauss_mD_sample, binning = TRUE, nh = 80, nb_bin_vect = c(40, 80))

# with a too small number of bins, the results are degenerated
bw.L2PCO(gauss_mD_sample, binning = TRUE, nh = 80, nb_bin_vect = c(5, 10))

bw.L2PCO(gauss_mD_sample, binning = TRUE, nh = 80, nb_bin_vect = c(40, 45))





