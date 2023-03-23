#' Compute the full PCO bandwith
#'
#' \code{bw.L2PCO} tries to minimise the PCO criterion (described and studied in Varet, S., Lacour, C., Massart, P., 
#' Rivoirard, V., (2019)) with a gold section search. For multivariate data, 
#' it searches for a full matrix.
#' 
#'
#' @param x_i the observations. Must be a matrix with d column and n lines (d the dimension and n the sample size)
#' @param nh the maximum number of PCO criterion evaluations during the golden section search. 
#' The default value is 40. The golden section search stop once this value is reached or 
#' if the tolerance is achieved, and return the middle of the interval.
#' @param K_name name of the kernel. Can be 'gaussian', 'epanechnikov', 
#' or 'biweight'. The default value is 'gaussian'.
#' @param binning default value is FALSE, that is the function computes the exact PCO criterion.
#' If set to TRUE allows to use binning. 
#' @param nb is the number of bins to use when binning is TRUE. 
#' For multivariate x_i, nb corresponds to the number of bins per dimension. The default value is 32.
#' @param tol is the maximum authorized length of the interval which contains the optimal h 
#' for univariate data. For multivariate data, it corresponds to the length of each hypercube axe.
#' The golden section search stop once this value is achieved or when nh is reached 
#' and return the middle of the interval. Its default value is 10^(-6).
#' @param adapt_nb_bin is a boolean used for univariate x_i. If set to TRUE, authorises the function to increase 
#' the number of bins if, with nb bins, the middle of the initial interval is not an admissible solution, that is 
#' if the criterion at the middle is greater than the mean of the criterion at the bounds of the initial interval of search.
#' @param nb_bin_vect can be set to have a different number of bins on each dimension
#'
#' @return a scalar for univariate data or a matrix for multivariate data corresponding to the optimal bandwidth according to the PCO criterion
#'
#' @references Varet, S., Lacour, C., Massart, P., Rivoirard, V., (2019). \emph{Numerical performance of Penalized
#' Comparison to Overfitting for multivariate kernel density estimation}. hal-02002275. \url{https://hal.archives-ouvertes.fr/hal-02002275}
#'
#' 
#' @seealso [stats::nrd0()], [stats::nrd()], [stats::ucv()], [stats::bcv()] and [stats::SJ()] 
#' for other univariate bandwidth selection and [stats::density()] to compute the associated density estimation.
#' @seealso [ks::Hlscv()], [ks::Hbcv()], [ks::ns()] for other multivariate bandwidth selection.
#' 
#' 
#' @examples 
#' 
#' # an example with simulated univariate data
#' 
#' # load univariate data
#' data("gauss_1D_sample")
#' 
#' # computes the optimal bandwith for the sample x_i with all parameters set to their default value 
#' bw.L2PCO(gauss_1D_sample)
#' 
#' 
#' # an example with simulated multivariate data
#' 
#' # load multivariate data
#' data("gauss_mD_sample")
#' 
#' # computes the optimal bandwith for the sample x_i with all parameters set to their default value 
#' # generates a warning since the tolerance value is not reached
#' bw.L2PCO(gauss_mD_sample)
#' 
#' # To avoid this warning, it is possible to increase the parameter nh
#' bw.L2PCO(gauss_mD_sample, nh = 80)
#' 
#' 
#'
#' @useDynLib PCObw
#' @importFrom Rcpp sourceCpp
#' @import RcppEigen
#' @export
bw.L2PCO <- function(x_i, nh = 40, K_name = 'gaussian', binning = FALSE, nb = 32, tol = 0.000001, adapt_nb_bin = FALSE, nb_bin_vect = NULL){
  
  
  
  
  
  
  if (!is.numeric(x_i)) {
    stop("invalid 'x_i'")
  }
  
  
  
  
  
  xi <- stats::na.omit(x_i)
  
  if (is.null(dim(xi))) {
    d <- 1
  }else{
    #d <- dim(xi)[[2]]
    d <- ncol(xi)
  }
  
  if (d == 1){
    n <- length(xi)
    if (is.na(n)) {
      stop("invalid length(x_i)")
    }
    if (is.na(nh) || nh <= 0){ 
      stop("invalid 'nh'")
    }
    
    if (binning){
      if (is.na(nb) || nb <= 0){ 
        stop("invalid 'nb'")
      }
      if (K_name == 'gaussian'){
        h_opt <- h_GK_1D_bin(xi = xi, nb_bin = nb, nh_max = nh, tol = tol, adapt_nb_bin = adapt_nb_bin)
      }else{
        if (K_name == 'epanechnikov'){
          h_opt <- h_EK_1D_bin(xi = xi, nb_bin = nb, nh_max = nh, tol = tol, adapt_nb_bin = adapt_nb_bin)
        }else{
          if (K_name == 'biweight'){
            h_opt <- h_BK_1D_bin(xi = xi, nb_bin = nb, nh_max = nh, tol = tol, adapt_nb_bin = adapt_nb_bin)
          }else{
            stop("This kernel has not been implemented")
          }
        }
      }
    }else{# binning = FALSE
      if (!binning){
        if (K_name == 'gaussian'){
          h_opt <- h_GK_1D_exact(xi = xi, nh_max = nh, tol = tol)
        }else{
          if (K_name == 'epanechnikov'){
            h_opt <- h_EK_1D_exact(xi = xi, nh_max = nh, tol = tol)
          }else{
            if (K_name == 'biweight'){
              h_opt <- h_BK_1D_exact(xi = xi, nh_max = nh, tol = tol)
            }else{
              stop("This kernel has not been implemented")
            }
          }
        }
      }else{
        stop("invalid 'binning'")
      }
    }
    return(h_opt)
  }else{ # d > 1
    
    
    n <- nrow(xi)
    if (is.na(n)) {
      stop("invalid nrow(x_i)")
    }
    
    if (is.na(nh) || nh <= 0){ 
      stop("invalid 'nh'")
    }
    
    if (binning){
      if (is.na(nb) || nb <= 0){ 
        stop("invalid 'nb'")
      }
      if (K_name == 'gaussian'){
        S <- array(data=NA, dim=c(d, d))
        S <- stats::cov(xi)
        if (is.null(nb_bin_vect)){
          h_opt <- h_GK_binned_mD_full(x_i = xi, S = S, nh_max = nh, tol = tol, nb_bin_per_axis = nb)
        }else{
          
          if (length(nb_bin_vect) != d){
            stop("invalid length for nb_bin_vect")
          }
          
          h_opt <- h_GK_binned_mD_full(x_i = xi, S = S, nh_max = nh, tol = tol, nb_bin_vect_ = round(abs(nb_bin_vect)))
          
        }
      }else{
        stop("This kernel has not been implemented")
      }
      
    }else{
      if (!binning){
        if (K_name == 'gaussian'){
          S <- array(data=NA, dim=c(d, d))
          S <- stats::cov(xi)
          h_opt <- h_GK_mD_full_exact(x_i = xi, S = S, nh_max = nh, tol = tol)
        }else{
          
              stop("This kernel has not been implemented")
        }
      }else{
        stop("invalid 'binning'")
      }
    }
    return(h_opt)
    
    
    
    
  }
  
  
}






















#' Compute the diagonal PCO bandwith
#'
#' \code{bw.L2PCO.diag} tries to minimise the PCO criterion (described and studied in Varet, S., Lacour, C., 
#' Massart, P., Rivoirard, V., (2019)) with a gold section search. For multivariate data, 
#' it searches for a diagonal matrix.
#' 
#'
#' @param x_i the observations. Must be a matrix with d column and n lines (d the dimension and n the sample size)
#' @param nh the maximum of possible bandwiths tested. 
#' The default value is 40.
#' @param K_name name of the kernel. Can be 'gaussian', 'epanechnikov', 
#' or 'biweight'. The default value is 'gaussian'.
#' @param binning can be set to TRUE or FALSE. 
#' The value TRUE allows to use binning. The default value FALSE
#' computes the exact PCO criterion.
#' @param nb is the number of bins to use when binning is TRUE. 
#' For multivariate x_i, nb corresponds to the number of bins per dimension. 
#' @param tol is the maximum authorized length of the interval which contains the optimal h 
#' for univariate data. For multivariate data, it corresponds to the length of each hypercube axe.
#' The golden section search stop once this value is achieved or when nh is reached 
#' and return the middle of the interval. Its default value is 10^(-6).
#' @param adapt_nb_bin is a boolean used for univariate x_i. If set to TRUE, authorises the function to increase 
#' the number of bins if, with nb bins, the middle of the initial interval is not an admissible solution, that is 
#' if the criterion at the middle is greater than the mean of the criterion at the bounds of the initial interval of search.
#' @param nb_bin_vect can be set to have a different number of bins on each dimension
#' 
#' @return a scalar for univariate data or a vector (the diagonal of the matrix) for multivariate data corresponding to the optimal bandwidth according to the PCO criterion
#'
#' @references Varet, S., Lacour, C., Massart, P., Rivoirard, V., (2019). \emph{Numerical performance of Penalized
#' Comparison to Overfitting for multivariate kernel density estimation}. hal-02002275. \url{https://hal.archives-ouvertes.fr/hal-02002275}
#'
#'
#' 
#' @seealso [stats::nrd0()], [stats::nrd()], [stats::ucv()], [stats::bcv()] and [stats::SJ()] 
#' for other univariate bandwidth selection and [stats::density()] to compute the associated density estimation.
#' @seealso [ks::Hlscv.diag()], [ks::Hbcv.diag()], [ks::ns.diag()] for other multivariate bandwidth selection.
#' 
#' 
#' 
#' 
#' 
#' @examples 
#' 
#' # an example with simulated univariate data
#' 
#' # load univariate data
#' data("gauss_1D_sample")
#' 
#' # computes the optimal bandwith for the sample x_i with all parameters set to their default value 
#' bw.L2PCO.diag(gauss_1D_sample)
#' 
#' 
#' # an example with simulated multivariate data
#' 
#' # load multivariate data
#' data("gauss_mD_sample")
#' 
#' # computes the optimal bandwith for the sample x_i with all parameters set to their default value 
#' # generates a warning since the tolerance value is not reached
#' bw.L2PCO.diag(gauss_mD_sample)
#' 
#' # To avoid this warning, it is possible to increase the parameter nh
#' bw.L2PCO.diag(gauss_mD_sample, nh = 80)
#' 
#' 
#' 
#' 
#'
#' @useDynLib PCObw
#' @importFrom Rcpp sourceCpp
#' @import RcppEigen
#' @export
bw.L2PCO.diag <- function(x_i, nh = 40, K_name = 'gaussian', binning = FALSE, nb = 32, tol = 0.000001, adapt_nb_bin = FALSE, nb_bin_vect = NULL){
  
  
  
  
  
  
    if (!is.numeric(x_i)) {
      stop("invalid 'x_i'")
    }
  
  
  
  xi <- stats::na.omit(x_i)
  
  if (is.null(dim(xi))) {
    d <- 1
  }else{
    #d <- dim(xi)[[2]]
    d <- ncol(xi)
  }
  
  
  
  if (d == 1){
    n <- length(xi)
    if (is.na(n)) {
      stop("invalid length(x_i)")
    }
    if (is.na(nh) || nh <= 0){ 
      stop("invalid 'nh'")
    }
    if (binning){
      if (is.na(nb) || nb <= 0){ 
        stop("invalid 'nb'")
      }
      if (K_name == 'gaussian'){
        h_opt <- h_GK_1D_bin(xi = xi, nb_bin = nb, nh_max = nh, tol = tol, adapt_nb_bin = adapt_nb_bin)
      }else{
        if (K_name == 'epanechnikov'){
          h_opt <- h_EK_1D_bin(xi = xi, nb_bin = nb, nh_max = nh, tol = tol, adapt_nb_bin = adapt_nb_bin)
        }else{
          if (K_name == 'biweight'){
            h_opt <- h_BK_1D_bin(xi = xi, nb_bin = nb, nh_max = nh, tol = tol, adapt_nb_bin = adapt_nb_bin)
          }else{
            stop("This kernel has not been implemented")
          }
        }
      }
      
    }else{
      if (!binning){
        if (K_name == 'gaussian'){
          h_opt <- h_GK_1D_exact(xi = xi, nh_max = nh, tol = tol)
        }else{
          if (K_name == 'epanechnikov'){
            h_opt <- h_EK_1D_exact(xi = xi, nh_max = nh, tol = tol)
          }else{
            if (K_name == 'biweight'){
              h_opt <- h_BK_1D_exact(xi = xi, nh_max = nh, tol = tol)
            }else{
              stop("This kernel has not been implemented")
            }
          }
        }
      }else{
        stop("invalid 'binning'")
      }
    }
    return(h_opt)
  }else{
    
    
    n <- nrow(xi)
    if (is.na(n)) {
      stop("invalid nrow(x_i)")
    }
    
    if (is.na(nh) || nh <= 0){ 
      stop("invalid 'nh'")
    }
    
    if (binning){
      if (is.na(nb) || nb <= 0){ 
        stop("invalid 'nb'")
      }else{
        if (K_name == 'gaussian'){
          
          if (is.null(nb_bin_vect)){
            h_opt <- h_GK_binned_mD_diag(x_i = xi, nh_max = nh, tol = tol, nb_bin_per_axis = nb)
          }else{
            if (length(nb_bin_vect) != d){
              stop("invalid length for nb_bin_vect")
            }
            h_opt <- h_GK_binned_mD_diag(x_i = xi, nh_max = nh, tol = tol, nb_bin_vect_ = round(abs(nb_bin_vect)))
          }
        }else{
          stop("This kernel has not been implemented")
        }
      }
    }else{
      if (!binning){
        if (K_name == 'gaussian'){
          
          h_opt <- h_GK_mD_diag_exact(x_i = xi, nh_max = nh, tol = tol)
          
        }else{
          
          stop("This kernel has not been implemented")
        }
      }else{
        stop("invalid 'binning'")
      }
    }
    
    return(diag(h_opt))
    
    
    
    
  }
  
  
}































.onUnload <- function (libpath) {
  library.dynam.unload("PCObw", libpath)
}