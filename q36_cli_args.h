#ifndef Q36_CLI_ARGS_H
#define Q36_CLI_ARGS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int q36_cli_token_is_option(const char *arg) {
    if (!arg || arg[0] != '-' || !arg[1]) return 0;
    /* Preserve negative numeric values used by penalties and steering. */
    return !((arg[1] >= '0' && arg[1] <= '9') || arg[1] == '.');
}

static inline int q36_cli_reject_option_arguments(int argc, char **argv,
                                                  const char *program) {
    for (int i = 1; i < argc; i++) {
        if (q36_cli_token_is_option(argv[i])) {
            fprintf(stderr, "%s: unknown option: %s\n", program, argv[i]);
            return 2;
        }
    }
    return 0;
}

static inline int q36_cli_require_no_arguments(int argc, char **argv,
                                               const char *program) {
    if (argc <= 1) return 0;
    if (q36_cli_token_is_option(argv[1]))
        fprintf(stderr, "%s: unknown option: %s\n", program, argv[1]);
    else
        fprintf(stderr, "%s: unexpected argument: %s\n", program, argv[1]);
    return 2;
}

/*
 * Expand GNU-style --option=value arguments before the existing, deliberately
 * small parsers inspect argv.  Keeping this here gives every executable the
 * same spelling behavior without replacing its option table or positional
 * argument rules.
 *
 * The allocation intentionally lives for the process lifetime.  Configuration
 * structs retain pointers into argv, so freeing it immediately after parsing
 * would make those pointers unsafe.
 */
static inline void q36_cli_expand_long_equals(int *argc, char ***argv,
                                              const char *program) {
    int old_argc = *argc;
    char **old_argv = *argv;
    char **expanded = calloc((size_t)old_argc * 2u + 1u, sizeof(*expanded));
    int out = 0;

    if (!expanded) {
        fprintf(stderr, "%s: out of memory while parsing options\n", program);
        exit(2);
    }

    for (int i = 0; i < old_argc; i++) {
        const char *arg = old_argv[i];
        const char *equal = NULL;
        if (i > 0 && arg && arg[0] == '-' && arg[1] == '-')
            equal = strchr(arg + 2, '=');
        if (!equal) {
            expanded[out++] = old_argv[i];
            continue;
        }
        if (equal == arg + 2) {
            fprintf(stderr, "%s: invalid option: %s\n", program, arg);
            exit(2);
        }
        if (!equal[1]) {
            fprintf(stderr, "%s: missing value for %.*s\n",
                    program, (int)(equal - arg), arg);
            exit(2);
        }

        size_t option_len = (size_t)(equal - arg);
        if ((option_len == strlen("--help") &&
             !strncmp(arg, "--help", option_len)) ||
            (option_len == strlen("--list") &&
             !strncmp(arg, "--list", option_len))) {
            fprintf(stderr, "%s: option does not take a value: %.*s\n",
                    program, (int)option_len, arg);
            exit(2);
        }

        char *option = malloc(option_len + 1u);
        if (!option) {
            fprintf(stderr, "%s: out of memory while parsing options\n", program);
            exit(2);
        }
        memcpy(option, arg, option_len);
        option[option_len] = '\0';
        expanded[out++] = option;
        expanded[out++] = (char *)(equal + 1);
    }

    expanded[out] = NULL;
    *argc = out;
    *argv = expanded;
}

#endif
