#!/usr/bin/env bash

# Fails if changelog_entry.yaml is empty or contains only whitespace
if [ ! -s changelog_entry.yaml ] || ! grep -q '[^[:space:]]' changelog_entry.yaml; then
  echo "changelog_entry.yaml is empty. Please add a changelog entry before merging."
  exit 1
fi
