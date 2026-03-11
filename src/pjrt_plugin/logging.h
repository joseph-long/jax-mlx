#pragma once

// ============================================================================
// Logging macros for PJRT plugin
//
// Log levels (set JAXPLUGIN_LOG_LEVEL before including this header):
//   0 = ERROR only
//   1 = ERROR + WARN (default)
//   2 = ERROR + WARN + INFO
//   3 = ERROR + WARN + INFO + DEBUG
// ============================================================================

#include <cstdio>

#ifndef JAXPLUGIN_LOG_LEVEL
#define JAXPLUGIN_LOG_LEVEL 1
#endif

// Level 0+: Errors (operation failures, invalid states)
#if JAXPLUGIN_LOG_LEVEL >= 0
#define JAXPLUGIN_LOG_ERROR(...)                           \
    do {                                                   \
        fprintf(stderr, "[JAXPLUGIN ERROR] " __VA_ARGS__); \
    } while (0)
#else
#define JAXPLUGIN_LOG_ERROR(...) \
    do {                         \
    } while (0)
#endif

// Level 1+: Warnings (unexpected but recoverable)
#if JAXPLUGIN_LOG_LEVEL >= 1
#define JAXPLUGIN_LOG_WARN(...)                           \
    do {                                                  \
        fprintf(stderr, "[JAXPLUGIN WARN] " __VA_ARGS__); \
    } while (0)
#else
#define JAXPLUGIN_LOG_WARN(...) \
    do {                        \
    } while (0)
#endif

// Level 2+: Info (operation flow, milestones)
#if JAXPLUGIN_LOG_LEVEL >= 2
#define JAXPLUGIN_LOG_INFO(...)                           \
    do {                                                  \
        fprintf(stderr, "[JAXPLUGIN INFO] " __VA_ARGS__); \
    } while (0)
#else
#define JAXPLUGIN_LOG_INFO(...) \
    do {                        \
    } while (0)
#endif

// Level 3+: Debug (detailed data, addresses, shapes)
#if JAXPLUGIN_LOG_LEVEL >= 3
#define JAXPLUGIN_LOG_DEBUG(...)                           \
    do {                                                   \
        fprintf(stderr, "[JAXPLUGIN DEBUG] " __VA_ARGS__); \
    } while (0)
#else
#define JAXPLUGIN_LOG_DEBUG(...) \
    do {                         \
    } while (0)
#endif
