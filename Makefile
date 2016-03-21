BUILD_DIR=build
LANDSLIDE_OPTIONS=--embed --linenos=table

# To install landslide: pip3 install --user landslide
LANDSLIDE=$(HOME)/.local/bin/landslide


all: createdir presentation

createdir:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/html
	mkdir -p $(BUILD_DIR)/pdf


Intro:
	$(LANDSLIDE) $(LANDSLIDE_OPTIONS) --destination=$(BUILD_DIR)/html/0_Intro.html 0_Intro/index.rst

Basics:
	$(LANDSLIDE) $(LANDSLIDE_OPTIONS) --destination=$(BUILD_DIR)/html/1_Basics.html 1_Basics/index.rst

presentation:
	$(LANDSLIDE) config.cfg



clean:
	rm -rf build
