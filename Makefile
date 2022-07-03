# Generate basic input files with carriage return instead of linefeed

MINI=$(wildcard test/*.mini)
CR=$(patsubst %.mini,%.CR,$(notdir $(MINI)))

all: $(CR)

%.CR: test/%.mini
	tr '\n' '\r' < $< > $@

clean:
	rm -f $(CR) README.html MININEC.INP

.PHONY: clean
