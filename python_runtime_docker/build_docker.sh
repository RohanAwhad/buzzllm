#!/bin/bash

build_python_exec() {
  docker build -t buzz/python-exec:latest -f Dockerfile .
}

clean_python_exec() {
  docker rmi buzz/python-exec:latest 2>/dev/null || true
  docker system prune -f
}

case "$1" in
  build-python-exec)
    build_python_exec
    ;;
  clean-python-exec)
    clean_python_exec
    ;;
  *)
    echo "Usage: $0 {build-python-exec|clean-python-exec}"
    exit 1
    ;;
esac
