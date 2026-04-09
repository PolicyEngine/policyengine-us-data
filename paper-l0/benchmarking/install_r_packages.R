packages <- c(
  "Matrix",
  "survey",
  "surveysd"
)

repos <- "https://cloud.r-project.org"

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (!length(missing)) {
  cat("All benchmarking R packages are already installed.\n")
  quit(save = "no", status = 0)
}

cat("Installing missing R packages:", paste(missing, collapse = ", "), "\n")
install.packages(missing, repos = repos)

failed <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE)]
if (length(failed)) {
  stop(sprintf("Failed to install: %s", paste(failed, collapse = ", ")))
}

cat("Benchmarking R packages installed successfully.\n")
