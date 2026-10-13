#pragma once
// bcore include-path shim: the vendored pow_v4.cpp includes
// <verification/pow_v3.h>; resolve it to the pow-utils pow_v3.h (the
// pow_v3 symbols pow_v4 uses are present in both copies). Never put logic
// here.
#include "../../pow_v3.h"
