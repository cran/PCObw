\name{PCObw-package}
\alias{PCObw-package}
\alias{PCObw}
\docType{package}
\title{
  Bandwidth Selector with Penalized Comparison to Overfitting Criterion
}
\description{
  Bandwidth selector according to the Penalised Comparison to Overfitting (P.C.O.) 
             criterion as described in Varet, S., Lacour, C., Massart, P., Rivoirard, V., (2019). 
             It can be used with univariate and multivariate data. 
}
\details{
  bw.L2PCO(x_i, ...)
  
  bw.L2PCO.diag(x_i, ...)
  
  select the optimal bandwith according to PCO criterion where x_i are the data (a numeric matrix or a numeric vector).
}
\author{
S. Varet.

Maintainer: S. Varet <suzanne.varet@universite-paris-saclay.fr>
}
\references{
  Varet, S., Lacour, C., Massart, P., Rivoirard, V., (2019). \emph{Numerical performance of Penalized Comparison to Overfitting for multivariate kernel density estimation}. hal-02002275. \href{https://hal.archives-ouvertes.fr/hal-02002275}{https://hal.archives-ouvertes.fr/hal-02002275}

}
\keyword{bandwidth}
\keyword{kernel density estimation}

\examples{

# load univariate data
data("gauss_1D_sample")
 
# computes the optimal bandwith for the sample x_i with all parameters set to their default value 
bw.L2PCO(gauss_1D_sample) 

  
}
