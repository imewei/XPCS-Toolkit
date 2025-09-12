#!/bin/bash

echo "$(date): Starting package preparation for XPCS Toolkit"
echo "$(date): Building source distribution..."
python -m build --sdist
if [ $? -eq 0 ]; then
    echo "$(date): Source distribution build successful"
else
    echo "$(date): Source distribution build failed"
    exit 1
fi

echo "$(date): Building wheel distribution..."
python -m build --wheel
if [ $? -eq 0 ]; then
    echo "$(date): Wheel distribution build successful"
else
    echo "$(date): Wheel distribution build failed"
    exit 1
fi

echo "$(date): Package preparation completed successfully"
