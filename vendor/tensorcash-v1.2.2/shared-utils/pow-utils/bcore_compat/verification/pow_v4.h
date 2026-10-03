#pragma once
// bcore include-path shim. pow_v4.{h,cpp} in shared-utils/pow-utils are
// byte-identical vendored copies of bcore src/verification/pow_v4.{h,cpp}
// and therefore include themselves as <verification/pow_v4.h>; this
// directory is added to the include path so that spelling resolves to the
// vendored file. Never put logic here.
#include "../../pow_v4.h"
