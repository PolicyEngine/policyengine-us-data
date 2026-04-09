script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Could not determine greg_runner.R path")
}
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
source(file.path(script_dir, "read_npy.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6L) {
  stop("Usage: greg_runner.R X.mtx target_metadata.csv initial_weights.npy output_weights.csv maxit epsilon")
}

matrix_path <- args[[1]]
target_csv <- args[[2]]
weights_npy <- args[[3]]
output_csv <- args[[4]]
maxit <- as.integer(args[[5]])
epsilon <- as.numeric(args[[6]])

library(Matrix)
library(survey)

X_targets_by_units <- readMM(matrix_path)
target_meta <- read.csv(target_csv, stringsAsFactors = FALSE)
base_weights <- read_npy_vector(weights_npy)
population <- as.numeric(target_meta$value)

mm <- Matrix::t(X_targets_by_units)
if (nrow(mm) != length(base_weights)) {
  stop("Unit count mismatch between matrix and initial weights")
}
if (ncol(mm) != length(population)) {
  stop("Target count mismatch between matrix and target metadata")
}

cal_linear <- get("cal.linear", envir = asNamespace("survey"))
g <- survey:::grake(
  mm = mm,
  ww = base_weights,
  calfun = cal_linear,
  bounds = list(lower = -Inf, upper = Inf),
  population = population,
  epsilon = epsilon,
  verbose = FALSE,
  maxit = maxit
)

fitted_weights <- as.numeric(base_weights * g)
write.csv(
  data.frame(unit_index = seq_along(fitted_weights) - 1L, fitted_weight = fitted_weights),
  output_csv,
  row.names = FALSE
)
