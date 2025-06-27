# Makefile for physics library
CC = clang
CFLAGS = -O3 -fPIC -march=native -fno-math-errno
LDFLAGS = -shared

# Detect OS for proper shared library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
    LDFLAGS += -dynamiclib
else
    LIB_EXT = .so
endif

TARGET = libphysics$(LIB_EXT)
SOURCES = physics_lib.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

test: $(TARGET)
	@echo "Library built successfully: $(TARGET)"

.PHONY: all clean test