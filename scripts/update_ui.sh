#!/bin/bash

WD=""../src/xpcs_toolkit""

echo "$(date): Starting UI file update for XPCS Toolkit"
echo "$(date): Working directory: $WD"

# ui file
echo "$(date): Converting UI file to Python..."
pyside6-uic $WD/ui/xpcs.ui -o viewer_ui.py
if [ $? -eq 0 ]; then
    echo "$(date): UI file conversion successful"
else
    echo "$(date): UI file conversion failed"
    exit 1
fi

# resource file goes to the current level
echo "$(date): Converting resource file..."
pyside6-rcc $WD/ui/resources/icons.qrc -o $WD/icons_rc.py
if [ $? -eq 0 ]; then
    echo "$(date): Resource file conversion successful"
else
    echo "$(date): Resource file conversion failed"
    exit 1
fi

echo "$(date): Updating import statements..."
sed 's/import icons_rc.*/from . import icons_rc/' viewer_ui.py > $WD/viewer_ui.py
if [ $? -eq 0 ]; then
    echo "$(date): Import statement update successful"
else
    echo "$(date): Import statement update failed"
    exit 1
fi

echo "$(date): Cleaning up temporary files..."
rm viewer_ui.py

echo "$(date): UI update completed successfully"
