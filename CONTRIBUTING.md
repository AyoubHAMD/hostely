# Contributing to hostely

Issues and PRs are welcome. Two constraints define this project — please
keep them in mind before opening a PR:

1. **Apple-only is the product.** hostely targets Apple Silicon on
   macOS 15+. Cross-platform abstractions, Linux ports, or CUDA anything
   are out of scope.
2. **No llama.cpp forks.** We link upstream `ggml-metal` exactly as it
   ships. If a fix belongs in llama.cpp, it goes to llama.cpp.

## Setup

```sh
git clone --recurse-submodules https://github.com/AyoubHAMD/hostely.git
cd hostely
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Apple clang (Xcode command-line tools) is required — the Objective-C++
Metal probe won't compile with other toolchains.

## Before you open a PR

```sh
cmake --build build -j && ./scripts/smoke.sh
```

A PR that hasn't passed the smoke script will be asked to run it. CI runs
the build and the smoke script on every PR, so push early and let it
verify.

## What's most wanted right now

- **Testing on other Apple Silicon configs** — M1/M2/M3, different RAM
  sizes, and especially *maxed-out* machines (report what the fit advisor
  gets wrong).
- **Model compatibility reports** — which GGUFs serve correctly, which
  hit the "tight" fit verdict, and with what `--ctx-size`.
- **Bugs in session KV reuse** — anything where a multi-turn conversation
  produces output that a cold stateless request doesn't.