# build the cython extension using GCC compiler

SHELL = /bin/sh
CC    = gcc-5

# ----- Build Targets ----- #
.PHONY: all list clean

all:
	python setup.py build_ext --inplace

# list all possible target in this makefile
list:
	@echo "LIST OF TARGETS:"
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
	| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
	| sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

# clean all temp files
clean:
	@echo "Clean up workbench"
	rm  -fv   *.tmp
	rm  -fv   tmp_*
	rm  -fv   *.geom
	rm  -fv   *.vtr
	rm  -fv   *.xml
	rm  -fv   *.gridata.pdf