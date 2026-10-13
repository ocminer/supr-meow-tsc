#pragma once
// bcore include-path shim. candidate_band.h in shared-utils/pow-utils is a
// byte-identical vendored copy of bcore src/verification/candidate_band.h
// and credit_v4.cpp includes it as <verification/candidate_band.h>; this
// directory is added to the include path so that spelling resolves to the
// vendored file. Never put logic here.
#include "../../candidate_band.h"
