BUILD_DIR=build

# To install landslide: pip3 install --user landslide
LANDSLIDE=$(HOME)/.local/bin/landslide


all: createdir Intro

createdir:
	mkdir -p $(BUILD_DIR)


Intro:
	$(LANDSLIDE) --embed --destination=$(BUILD_DIR)/0_Intro.html 0_Intro/index.rst


clean:
	rm -rf build
