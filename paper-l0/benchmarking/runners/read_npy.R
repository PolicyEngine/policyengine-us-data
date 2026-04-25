read_npy_vector <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)

  magic <- readBin(con, what = "raw", n = 6, endian = "little")
  if (!identical(as.integer(magic), as.integer(charToRaw("\x93NUMPY")))) {
    stop("Unsupported .npy file: bad magic header")
  }

  major <- readBin(con, what = "integer", n = 1, size = 1, signed = FALSE)
  minor <- readBin(con, what = "integer", n = 1, size = 1, signed = FALSE)

  header_len <- if (major == 1L) {
    readBin(con, what = "integer", n = 1, size = 2, signed = FALSE, endian = "little")
  } else if (major %in% c(2L, 3L)) {
    readBin(con, what = "integer", n = 1, size = 4, signed = FALSE, endian = "little")
  } else {
    stop("Unsupported .npy version")
  }

  header_raw <- readBin(con, what = "raw", n = header_len, endian = "little")
  header <- rawToChar(header_raw)

  descr_match <- regmatches(header, regexpr("'descr': *'[^']+'", header))
  shape_match <- regmatches(header, regexpr("'shape': *\\([^\\)]*\\)", header))
  fortran_match <- regmatches(header, regexpr("'fortran_order': *(True|False)", header))

  if (length(descr_match) == 0 || length(shape_match) == 0 || length(fortran_match) == 0) {
    stop("Could not parse .npy header")
  }

  descr <- sub("^'descr': *'([^']+)'$", "\\1", descr_match)
  fortran_order <- sub("^'fortran_order': *(True|False)$", "\\1", fortran_match)
  if (fortran_order != "False") {
    stop("Only C-order .npy arrays are supported")
  }

  shape_text <- sub("^'shape': *\\(([^\\)]*)\\)$", "\\1", shape_match)
  shape_parts <- trimws(unlist(strsplit(shape_text, ",")))
  shape_parts <- shape_parts[nzchar(shape_parts)]
  dims <- as.integer(shape_parts)

  if (length(dims) != 1L || is.na(dims[1])) {
    stop("Only 1D .npy vectors are supported")
  }

  n <- dims[1]

  if (descr == "<f8") {
    return(readBin(con, what = "double", n = n, size = 8, endian = "little"))
  }
  if (descr == "<f4") {
    return(as.numeric(readBin(con, what = "numeric", n = n, size = 4, endian = "little")))
  }
  if (descr == "<i8") {
    return(as.numeric(readBin(con, what = "integer", n = n, size = 8, endian = "little")))
  }
  if (descr == "<i4") {
    return(as.numeric(readBin(con, what = "integer", n = n, size = 4, endian = "little")))
  }
  if (descr == "|b1") {
    return(as.logical(readBin(con, what = "integer", n = n, size = 1, signed = FALSE)))
  }

  stop(sprintf("Unsupported .npy dtype: %s", descr))
}
