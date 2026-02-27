CXX      := g++
STD      := -std=c++17
OPTS     := -O3 -march=native -fvisibility=hidden -fPIC -DNDEBUG
WARN     := -Wall -Wextra -Wno-unused-parameter
PY_INC   := $(shell python3 -m pybind11 --includes)
PY_EXT   := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
ROLLING_OUT := features/_quant_rolling$(PY_EXT)

.PHONY: all rolling clean

all: rolling

rolling: $(ROLLING_OUT)

$(ROLLING_OUT): ext/rolling/bindings.cpp ext/rolling/rolling_window.hpp ext/rolling/technical.hpp
	$(CXX) $(STD) $(OPTS) $(WARN) $(PY_INC) -Iext/rolling -shared -o $@ $<

clean:
	rm -f features/_quant_rolling*.so
