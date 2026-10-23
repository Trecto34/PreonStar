#ifndef P36_WEB_H
#define P36_WEB_H

#include <stddef.h>
#include <stdbool.h>

typedef int (*p36_web_confirm_fn)(void *privdata, const char *message,
                                  char *err, size_t err_len);
typedef void (*p36_web_log_fn)(void *privdata, const char *message);
typedef bool (*p36_web_cancel_fn)(void *privdata);

typedef struct {
    const char *home_dir;
    int port;
    p36_web_confirm_fn confirm;
    void *confirm_privdata;
    p36_web_log_fn log;
    void *log_privdata;
    p36_web_cancel_fn cancel;
    void *cancel_privdata;
} p36_web_config;

typedef struct p36_web p36_web;

p36_web *p36_web_create(const p36_web_config *cfg);
void p36_web_free(p36_web *web);

char *p36_web_google_search(p36_web *web, const char *query,
                            char *err, size_t err_len);
char *p36_web_visit_page(p36_web *web, const char *url,
                         char *err, size_t err_len);

#endif
