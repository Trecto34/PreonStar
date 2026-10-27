CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wshadow -Wno-unused-function
LDFLAGS ?=
LDLIBS ?= -lm -pthread

all: qwen36-quantize mtp-head-extract

qwen36-quantize: qwen36-quantize.c quants.c quants.h ../p36_cli_args.h
	$(CC) $(CFLAGS) -o $@ qwen36-quantize.c quants.c $(LDFLAGS) $(LDLIBS)

mtp-head-extract: mtp-head-extract.c quants.c quants.h
	$(CC) $(CFLAGS) -o $@ mtp-head-extract.c quants.c $(LDFLAGS) $(LDLIBS)

clean:
	rm -f qwen36-quantize mtp-head-extract

.PHONY: all clean
