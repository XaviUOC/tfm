#!/bin/bash
function show_help {
    echo Usage: runevery.sh -t [time] [command]
}
OPTIND=1
time=1
while getopts "h?t:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    t)  time=$OPTARG
        ;;
    esac
done
shift $((OPTIND-1))

while true; do
    $@ &
    sleep $time
    pid=$!
    kill -9 $pid
done
