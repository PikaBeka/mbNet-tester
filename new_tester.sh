#!/bin/bash

while getopts m: flag
do
    case "${flag}" in
        m) method="${OPTARG}";;
    esac
done

echo "Method: $method";