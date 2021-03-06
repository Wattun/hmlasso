% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plotter.R
\name{plot.hmlasso}
\alias{plot.hmlasso}
\title{Plot a solution path}
\usage{
\method{plot}{hmlasso}(x, xlim = NULL, ylim = NULL, ...)
}
\arguments{
\item{x}{hmlasso model}

\item{xlim}{x range}

\item{ylim}{y range}

\item{...}{parameters of matlines function}
}
\description{
Plot a solution path
}
\examples{
X_incompl <- as.matrix(iris[, 1:3])
X_incompl[1:5,1] <- NA
X_incompl[6:10,2] <- NA
y <- iris[, 4]
fit <- hmlasso(X_incompl, y, nlambda=50, lambda.min.ratio=1e-2)
plot(fit)

}
