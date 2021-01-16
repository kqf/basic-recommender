
all: data/*.csv
	python model/develop.py

.PHONY: all
