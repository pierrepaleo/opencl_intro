#!/bin/bash

SLIDES="0_Introduction 1_Project 2_Distribution 3_Test 4_Documentation"
BUILD_DIR="build"
LANDSLIDE=/home/paleo/.local/bin/landslide

pushd $( dirname $0 )

mkdir -p ${BUILD_DIR}

for DIR in ${SLIDES}; do
    pushd "${DIR}"
    echo "Build ${DIR}"
    $LANDSLIDE --embed --destination=../${BUILD_DIR}/${DIR}.html index.rst
    popd
done

popd
