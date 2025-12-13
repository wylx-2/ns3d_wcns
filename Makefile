
# Allow overriding compiler from environment
CXX = mpicxx

# Compiler flags: optimization, C++17, warnings, include current dir.
# Also generate dependency files (-MMD -MP)
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -I. -MMD -MP
LDFLAGS =

# Local FFTW (with MPI) install prefix. Override externally if needed.
FFTW_PREFIX ?= /home/huangwz/third_party/fftw-mpi

# Add FFTW include/lib paths and link flags (rpath so runtime can find libs).
CXXFLAGS += -I$(FFTW_PREFIX)/include
LDFLAGS += -L$(FFTW_PREFIX)/lib -lfftw3_mpi -lfftw3 -lm -Wl,-rpath,$(FFTW_PREFIX)/lib

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
DEPS = $(OBJS:.o=.d)

TARGET = ns3d

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Include dependency files if they exist
-include $(DEPS)

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

run: $(TARGET)
	@echo "Run with mpirun, e.g.: mpirun -n 4 ./$(TARGET)"
	mpirun -n 4 ./$(TARGET)
