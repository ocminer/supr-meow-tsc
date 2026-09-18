#pragma once

#ifdef _WIN32
#include <cstdlib>
#include <io.h>
inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name)) return 0;
    return _putenv_s(name, value);
}
inline int isatty(int fd) { return _isatty(fd); }
#else
#include <unistd.h>
#endif
