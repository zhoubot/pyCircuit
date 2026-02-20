.PHONY: help configure tools smoke install package clean

BUILD_DIR ?= build

help:
	@echo "Targets:"
	@echo "  configure  Configure CMake (needs LLVM_DIR/MLIR_DIR)"
	@echo "  tools      Build pycc + pyc-opt"
	@echo "  smoke      Run compiler + simulation smoke checks"
	@echo "  install    Install into dist/pycircuit/"
	@echo "  package    Build a TGZ via CPack"
	@echo "  clean      Remove build/ and dist/"

configure:
	@if [ -z "$$LLVM_DIR" ] || [ -z "$$MLIR_DIR" ]; then \
	  echo "error: set LLVM_DIR and MLIR_DIR"; \
	  exit 2; \
	fi
	cmake -G Ninja -S . -B "$(BUILD_DIR)" \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DLLVM_DIR="$$LLVM_DIR" \
	  -DMLIR_DIR="$$MLIR_DIR"

tools: configure
	ninja -C "$(BUILD_DIR)" pycc
	ninja -C "$(BUILD_DIR)" pyc-opt 2>/dev/null || true

smoke: tools
	PYCC="$(BUILD_DIR)/bin/pycc" bash flows/scripts/run_examples.sh
	PYCC="$(BUILD_DIR)/bin/pycc" bash flows/scripts/run_sims.sh

install: tools
	cmake --install "$(BUILD_DIR)" --prefix dist/pycircuit

package: tools
	(cd "$(BUILD_DIR)" && cpack -G TGZ)

clean:
	rm -rf "$(BUILD_DIR)" dist

