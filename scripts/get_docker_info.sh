#!/bin/sh

version_only=
if [ "$1" = "-v" ]; then
  version_only=1
  shift 1
fi

dockerfile=$1
shift 1

if [ ! -f  $dockerfile ]; then
  echo "Dockerfile not found" >> /dev/stderr
  exit 1
fi

if [ -z "$1"  ]; then
  echo "No target specified" >> /dev/stderr
  exit 1
fi

tag=$(grep "AS $1\$" $dockerfile | sed -e 's/FROM *//' -e 's/ *AS .*//')
if [ -z "$tag" ]; then
  echo "Target $1 not defined" >> /dev/stderr
  exit 1
fi
if [ "$version_only" = "1" ]; then
   echo $tag | sed -e 's/.*://'
else
   echo $tag
fi
