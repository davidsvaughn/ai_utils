#!/bin/bash
P=$1
X1=$2
X2=$3

cd "$P"

for name in *$X1; do
    newname="$(echo "$name" | sed "s/$X1/$X2/g")"
    echo "$name --> $newname"
    mv "$name" "$newname"
done

cd ..