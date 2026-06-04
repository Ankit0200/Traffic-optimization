#!/bin/bash
# Build the SUMO network from node + edge files
# Run this once, or whenever you change .nod.xml / .edg.xml

netconvert \
    --node-files=intersection.nod.xml \
    --edge-files=intersection.edg.xml \
    --output-file=intersection.net.xml \
    --tls.guess true

echo "Network built: intersection.net.xml"
echo "Run with:  sumo-gui -c intersection.sumocfg"
