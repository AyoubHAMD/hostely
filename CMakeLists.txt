cmake_minimum_required(VERSION 3.25)
# llama.cpp's CMakeLists.txt sets `cmake_minimum_required(VERSION 3.14...3.28)`
# (the trailing 3.28 caps the maximum CMake version it accepts). Pinning
# CMAKE_POLICY_VERSION_MINIMUM to 3.28 keeps us under that ceiling for any
# add_subdirectory() we pull in. This is what the CMake 4 release notes
# recommend for projects that need to consume 3.x subprojects.
set(CMAKE_POLICY_VERSION_MINIMUM 3.28)
project(hostely
    VERSION 0.1.0
    DESCRIPTION "Self-hosting manager for Apple Silicon (containers + Metal LLM inference)"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Toolchain expectations
# -----------------------------------------------------------------------------
# - Apple Silicon only (arm64); x86_64 Mac builds for testing only.
# - macOS 15.0 floor
# - C++20
# -----------------------------------------------------------------------------

if(NOT APPLE)
    message(FATAL_ERROR "hostely only builds on macOS.")
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(WARNING
        "hostely targets Apple Silicon. You're on x86_64; "
        "Metal-backed inference will not work. Continuing anyway.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HOSTELY_MACOS_MIN "15.0")
set(CMAKE_OSX_DEPLOYMENT_TARGET "${HOSTELY_MACOS_MIN}")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter        # CLI handlers often have unused args; tighten later
    )
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

message(STATUS "hostely ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  target OS  : macOS ${HOSTELY_MACOS_MIN}+")
message(STATUS "  arch       : ${CMAKE_SYSTEM_PROCESSOR}")

# -----------------------------------------------------------------------------
# Third-party deps via FetchContent
# -----------------------------------------------------------------------------
include(FetchContent)

# toml++ — header-only TOML parser/writer.
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tomlplusplus)

# cpp-httplib — single-header HTTP server/client. Header-only.
FetchContent_Declare(
    cpp-httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.5
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cpp-httplib)

# nlohmann/json — JSON for OpenAI-compatible request/response bodies.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# -----------------------------------------------------------------------------
# llama.cpp (vendored as a git submodule under third_party/llama.cpp)
# -----------------------------------------------------------------------------
# We consume llama.cpp via add_subdirectory. We disable everything we don't
# need (tests, examples, server, app, openssl) to keep the build time and
# surface area tight. ggml-metal stays ON by default on Apple Silicon.
# -----------------------------------------------------------------------------
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

# ggml options we explicitly want.
set(GGML_METAL                ON  CACHE BOOL "" FORCE)   # default on Apple; explicit for clarity
set(GGML_METAL_NDEBUG         ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE               OFF CACHE BOOL "" FORCE)   # no -march=native; portability
set(GGML_CPU_ALL_VARIANTS     OFF CACHE BOOL "" FORCE)   # don't compile every ISA variant

add_subdirectory(third_party/llama.cpp ${CMAKE_BINARY_DIR}/_llama_build)

# -----------------------------------------------------------------------------
# OpenSSL 3 — TLS for the exposure stack (proxy, ACME, tunnel). Optional: the
# container/serve features build without it; exposure commands report it.
# -----------------------------------------------------------------------------
find_package(OpenSSL 3 QUIET)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL   : ${OPENSSL_VERSION} (${OPENSSL_ROOT_DIR})")
else()
    message(STATUS "OpenSSL   : NOT FOUND — exposure features (proxy/certs/tunnel) disabled")
endif()

# -----------------------------------------------------------------------------
# hostely core library
# -----------------------------------------------------------------------------

add_library(hostely_core STATIC
    src/apps/app.cpp
    src/apps/compose.cpp
    src/cli/args.cpp
    src/cli/exposure_cli.cpp
    src/config/config.cpp
    src/inference/lockfile.cpp
    src/inference/server.cpp
    src/inference/session.cpp
    src/log/logger.cpp
    src/models/fit.cpp
    src/models/pull.cpp
    src/models/registry.cpp
    src/paths.cpp
    src/resources/limits.cpp
    src/resources/system.cpp
    src/resources/metal_probe.mm
    src/services/cli.cpp
    src/services/manager.cpp
    src/top/sample.cpp
    src/top/render.cpp
    src/top/top.cpp
    src/top/tui.cpp
)

if(OpenSSL_FOUND)
    target_sources(hostely_core PRIVATE
        src/exposure/certs.cpp
        src/exposure/dns.cpp
        src/exposure/acme.cpp
        src/exposure/expose.cpp
        src/proxy/proxy.cpp
        src/tunnel/yamux.cpp
        src/tunnel/client.cpp
    )
    target_compile_definitions(hostely_core PUBLIC HOSTELY_HAVE_OPENSSL=1)
    target_link_libraries(hostely_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

# The single .mm file compiles as Obj-C++; the rest of the library is C++.
set_source_files_properties(src/resources/metal_probe.mm
    PROPERTIES
        COMPILE_FLAGS "-x objective-c++"
)

target_include_directories(hostely_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hostely_core
    PUBLIC
        # Foundation is needed for NSError/NSString in any .mm TU we add later.
        "-framework Foundation"
        tomlplusplus::tomlplusplus
    PRIVATE
        httplib::httplib          # cpp-httplib, only inference uses it
        nlohmann_json::nlohmann_json
        llama                     # llama.cpp
        ggml                      # ggml (transitive via llama but be explicit)
        ggml::ggml                # alias
)

# The Metal probe needs Metal linked into the .mm TU. Linking against the
# whole library is the simplest way; it's a no-op on non-Apple.
target_link_libraries(hostely_core PRIVATE "-framework Metal")

# -----------------------------------------------------------------------------
# hostely binary
# -----------------------------------------------------------------------------

add_executable(hostely
    src/main.cpp
)

target_link_libraries(hostely
    PRIVATE
        hostely_core
)

target_compile_definitions(hostely
    PRIVATE
        HOSTELY_VERSION="${PROJECT_VERSION}"
)

# -----------------------------------------------------------------------------
# hostely-relay — standalone public server for `hostely tunnel` (runs on a VPS)
# -----------------------------------------------------------------------------
if(OpenSSL_FOUND)
    add_executable(hostely-relay
        src/tunnel/relay.cpp
        src/tunnel/yamux.cpp
    )
    target_compile_features(hostely-relay PRIVATE cxx_std_20)
    target_include_directories(hostely-relay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(hostely-relay PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    install(TARGETS hostely-relay RUNTIME DESTINATION bin)
endif()

# -----------------------------------------------------------------------------
# Install rules (placeholder; no Homebrew formula yet)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS hostely RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
