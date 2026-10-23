#ifndef P36_PROMPT_PREFIX_H
#define P36_PROMPT_PREFIX_H

#include "p36.h"

#include <stddef.h>

typedef enum {
    P36_PROMPT_PREFIX_USER,
    P36_PROMPT_PREFIX_ASSISTANT,
} p36_prompt_prefix_role;

typedef struct {
    p36_prompt_prefix_role role;
    char *content;
} p36_prompt_prefix_turn;

typedef struct {
    p36_prompt_prefix_turn *turns;
    size_t count;
} p36_prompt_prefix;

int p36_prompt_prefix_parse(p36_prompt_prefix *out,
                            const char *data,
                            size_t len,
                            char *error,
                            size_t error_cap);
int p36_prompt_prefix_load(p36_prompt_prefix *out,
                           const char *path,
                           char *error,
                           size_t error_cap);
void p36_prompt_prefix_free(p36_prompt_prefix *prefix);

static inline void p36_prompt_prefix_append(p36_engine *engine,
                                             p36_tokens *tokens,
                                             const p36_prompt_prefix *prefix) {
    if (!prefix) return;
    for (size_t i = 0; i < prefix->count; i++) {
        const p36_prompt_prefix_turn *turn = &prefix->turns[i];
        const bool assistant = turn->role == P36_PROMPT_PREFIX_ASSISTANT;
        p36_chat_append_message(engine, tokens,
                                assistant ? "assistant" : "user",
                                turn->content);
    }
}

#endif
