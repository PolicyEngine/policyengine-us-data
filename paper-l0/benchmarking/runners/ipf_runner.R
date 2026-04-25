script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Could not determine ipf_runner.R path")
}
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
source(file.path(script_dir, "read_npy.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10L) {
  stop("Usage: ipf_runner.R unit_metadata.csv ipf_target_metadata.csv initial_weights.npy output_weights.csv max_iter bound epsP epsH household_id_col weight_col")
}

unit_csv <- args[[1]]
target_csv <- args[[2]]
weights_npy <- args[[3]]
output_csv <- args[[4]]
max_iter <- as.integer(args[[5]])
bound <- as.numeric(args[[6]])
epsP <- as.numeric(args[[7]])
epsH <- as.numeric(args[[8]])
household_id_col <- args[[9]]
weight_col <- args[[10]]

if (!requireNamespace("surveysd", quietly = TRUE)) {
  stop("The surveysd package is required for IPF benchmarks")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("The data.table package is required for IPF benchmarks")
}

build_margin_array <- function(df) {
  variables <- strsplit(df$variables[[1]], "\\|")[[1]]
  rows <- lapply(seq_len(nrow(df)), function(i) {
    cell <- df$cell[[i]]
    parts <- strsplit(cell, "\\|")[[1]]
    entries <- strsplit(parts, "=")
    row <- as.list(setNames(vapply(entries, `[`, "", 2L), vapply(entries, `[`, "", 1L)))
    row$Freq <- as.numeric(df$target_value[[i]])
    as.data.frame(row, stringsAsFactors = FALSE)
  })
  frame <- do.call(rbind, rows)
  stats::xtabs(
    stats::as.formula(paste("Freq ~", paste(variables, collapse = " + "))),
    data = frame
  )
}

unit_data <- read.csv(unit_csv, stringsAsFactors = FALSE)
target_meta <- read.csv(target_csv, stringsAsFactors = FALSE)
base_weights <- read_npy_vector(weights_npy)
unit_data <- data.table::as.data.table(unit_data)

if (!(weight_col %in% names(unit_data))) {
  if (!("unit_index" %in% names(unit_data))) {
    stop("Unit metadata must include either base weights or a unit_index column")
  }
  if (max(unit_data$unit_index) >= length(base_weights)) {
    stop("unit_index contains values outside the initial weight vector")
  }
  unit_data[[weight_col]] <- base_weights[unit_data$unit_index + 1L]
}

if (!("target_type" %in% names(target_meta))) {
  stop("ipf target metadata must include a target_type column")
}
unsupported_types <- setdiff(unique(target_meta$target_type), "categorical_margin")
if (length(unsupported_types) > 0L) {
  stop(sprintf(
    "ipf_runner.R only supports target_type='categorical_margin'; got: %s",
    paste(unsupported_types, collapse = ", ")
  ))
}

conP <- list()
conH <- list()
margin_rows_all <- target_meta
for (margin_id in unique(margin_rows_all$margin_id)) {
  margin_rows <- margin_rows_all[margin_rows_all$margin_id == margin_id, , drop = FALSE]
  margin_array <- build_margin_array(margin_rows)
  scope <- unique(margin_rows$scope)
  if (length(scope) != 1L) {
    stop(sprintf("Margin %s has inconsistent scope values", margin_id))
  }
  if (scope[[1]] == "person") {
    conP[[length(conP) + 1L]] <- margin_array
  } else if (scope[[1]] == "household") {
    conH[[length(conH) + 1L]] <- margin_array
  } else {
    stop(sprintf("Unsupported margin scope: %s", scope[[1]]))
  }
}

ipf_result <- surveysd::ipf(
  dat = unit_data,
  hid = if (household_id_col %in% names(unit_data)) household_id_col else NULL,
  conP = if (length(conP)) conP else NULL,
  conH = if (length(conH)) conH else NULL,
  epsP = epsP,
  epsH = epsH,
  verbose = FALSE,
  w = weight_col,
  bound = bound,
  maxIter = max_iter,
  meanHH = TRUE,
  returnNA = TRUE,
  nameCalibWeight = "calibWeight"
)

if (!("calibWeight" %in% names(ipf_result))) {
  stop("surveysd::ipf did not return a calibWeight column")
}

write.csv(
  data.frame(
    unit_index = if ("unit_index" %in% names(ipf_result)) ipf_result$unit_index else seq_len(nrow(ipf_result)) - 1L,
    fitted_weight = as.numeric(ipf_result$calibWeight)
  ),
  output_csv,
  row.names = FALSE
)
