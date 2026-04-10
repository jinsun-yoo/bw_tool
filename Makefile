CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pthread -I src
LDFLAGS  := -pthread

SRCS := src/main.cpp src/sampler.cpp src/writer.cpp
OBJS := $(SRCS:src/%.cpp=build/%.o)
BIN  := build/bw_monitor

PREFIX ?= $(HOME)

.PHONY: all clean install

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build:
	mkdir -p build

# Install binary and scripts to ~/bin (no sudo required)
install: all
	@mkdir -p "$(PREFIX)/bin"
	cp $(BIN) "$(PREFIX)/bin/bw_monitor"
	cp scripts/bw-start "$(PREFIX)/bin/bw-start"
	cp scripts/bw-stop  "$(PREFIX)/bin/bw-stop"
	chmod +x "$(PREFIX)/bin/bw-start" "$(PREFIX)/bin/bw-stop"
	@echo "Installed to $(PREFIX)/bin. Make sure $(PREFIX)/bin is in your PATH."

clean:
	rm -rf build
