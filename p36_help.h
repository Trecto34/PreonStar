#ifndef P36_HELP_H
#define P36_HELP_H

#include <stdbool.h>
#include <stdio.h>

typedef enum {
    P36_HELP_P36,
    P36_HELP_SERVER,
    P36_HELP_AGENT,
    P36_HELP_BENCH,
    P36_HELP_EVAL,
} p36_help_tool;

bool p36_help_topic_valid(p36_help_tool tool, const char *topic);
void p36_help_print(FILE *fp, p36_help_tool tool, const char *topic);

#endif
