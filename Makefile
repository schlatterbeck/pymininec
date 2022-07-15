# Generate basic input files with carriage return instead of linefeed

MINI=$(wildcard test/*.mini)
CR=$(patsubst %.mini,%.CR,$(notdir $(MINI)))

all: $(CR) tests

%.CR: test/%.mini
	tr '\n' '\r' < $< > $@

coverage:
	python3 -m pytest --cov-report term-missing --cov mininec test

tests:
	python3 -m pytest test

clean:
	rm -f $(CR) README.html MININEC.INP

.PHONY: clean tests coverage
