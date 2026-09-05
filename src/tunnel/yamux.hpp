#pragma once
// In-tree yamux-compatible multiplexer (streamable transport over one TCP
// connection). Wire format (12-byte header, big-endian):
//   version(1) type(1) flags(2) streamID(4) length(4)
// Types: Data(0), WindowUpdate(1), Ping(2), GoAway(3).
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace hostely::tunnel {

constexpr uint8_t kYamuxVersion = 0;
constexpr uint8_t kTypeData = 0, kTypeWindowUpdate = 1, kTypePing = 2,
                  kTypeGoAway = 3;
constexpr uint16_t kFlagSyn = 1, kFlagAck = 2, kFlagFin = 4, kFlagRst = 8;
constexpr uint32_t kInitialWindow = 256 * 1024;

// A single multiplexed stream over a socket fd.
class MuxStream {
public:
    MuxStream(class Mux* mux, uint32_t id) : mux_(mux), id_(id) {}
    ~MuxStream();
    int read(uint8_t* buf, size_t len);      // blocks
    // Non-blocking: returns -1 when no buffered data (never blocks).
    int try_read(uint8_t* buf, size_t len);
    bool write(const uint8_t* buf, size_t len);
    void close();
    uint32_t id() const { return id_; }
    bool closed() const { return closed_; }
    void push_data(const uint8_t* d, size_t n);   // from Mux read loop

private:
    class Mux* mux_;
    uint32_t id_;
    std::mutex mu_;
    std::string buffer_;        // pending received data
    bool closed_ = false;
    friend class Mux;
};

// One multiplexed session over a socket fd (fd ownership stays with caller).
class Mux {
public:
    Mux(int fd, bool client);
    ~Mux();
    std::shared_ptr<MuxStream> open_stream();         // client side: SYN
    std::shared_ptr<MuxStream> accept_stream();       // blocks; server side
    void close();
    bool ok() const { return !dead_; }

private:
    friend class MuxStream;
    bool send_frame(uint8_t type, uint16_t flags, uint32_t stream_id,
                    const uint8_t* payload, size_t len);
    void read_loop();

    int fd_;
    bool client_;
    std::mutex send_mu_;
    std::mutex mu_;
    std::map<uint32_t, std::shared_ptr<MuxStream>> streams_;
    uint32_t next_id_ = 1;      // odd for client, even for server
    std::shared_ptr<MuxStream> pending_accept_;
    std::thread reader_;
    bool dead_ = false;
};

}  // namespace hostely::tunnel