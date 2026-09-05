#include "tunnel/yamux.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace hostely::tunnel {

namespace {
void put_u16(uint8_t* p, uint16_t v) { *reinterpret_cast<uint16_t*>(p) = htons(v); }
void put_u32(uint8_t* p, uint32_t v) { *reinterpret_cast<uint32_t*>(p) = htonl(v); }
uint16_t get_u16(const uint8_t* p) { return ntohs(*reinterpret_cast<const uint16_t*>(p)); }
uint32_t get_u32(const uint8_t* p) { return ntohl(*reinterpret_cast<const uint32_t*>(p)); }
constexpr size_t kHeaderSize = 12;
constexpr uint16_t kDataFrameCap = 16 * 1024;
}  // namespace

// ---------------------------------------------------------------------------
// MuxStream
// ---------------------------------------------------------------------------

MuxStream::~MuxStream() { close(); }

void MuxStream::push_data(const uint8_t* d, size_t n) {
    std::lock_guard lk(mu_);
    buffer_.append(reinterpret_cast<const char*>(d), n);
}

bool MuxStream::write(const uint8_t* buf, size_t len) {
    while (len > 0) {
        size_t chunk = len < kDataFrameCap ? len : kDataFrameCap;
        {
            std::lock_guard lk(mu_);
            if (closed_) return false;
        }
        if (!mux_->send_frame(kTypeData, 0, id_, buf, chunk)) return false;
        buf += chunk;
        len -= chunk;
    }
    return true;
}

int MuxStream::try_read(uint8_t* buf, size_t len) {
    std::lock_guard lk(mu_);
    if (buffer_.empty()) return closed_ ? 0 : -1;
    size_t n = len < buffer_.size() ? len : buffer_.size();
    memcpy(buf, buffer_.data(), n);
    buffer_.erase(0, n);
    return static_cast<int>(n);
}

int MuxStream::read(uint8_t* buf, size_t len) {
    std::unique_lock lk(mu_);
    while (buffer_.empty()) {
        if (closed_) return 0;
        // Poll for more data via the mux read loop (data arrives through
        // push_data on the mux's reader thread).
        lk.unlock();
        // The mux read loop owns the socket; just yield briefly.
        usleep(2000);
        lk.lock();
    }
    size_t n = len < buffer_.size() ? len : buffer_.size();
    memcpy(buf, buffer_.data(), n);
    buffer_.erase(0, n);
    return static_cast<int>(n);
}

void MuxStream::close() {
    bool was;
    {
        std::lock_guard lk(mu_);
        was = closed_;
        closed_ = true;
    }
    if (!was && mux_) {
        mux_->send_frame(kTypeWindowUpdate, kFlagFin, id_, nullptr, 0);
    }
}

// ---------------------------------------------------------------------------
// Mux
// ---------------------------------------------------------------------------

Mux::Mux(int fd, bool client) : fd_(fd), client_(client) {
    next_id_ = client ? 1 : 2;
    reader_ = std::thread([this] { read_loop(); });
}

Mux::~Mux() { close(); }

void Mux::close() {
    {
        std::lock_guard lk(mu_);
        if (dead_) return;
        dead_ = true;
    }
    shutdown(fd_, SHUT_RDWR);
    if (reader_.joinable()) reader_.join();
    std::lock_guard lk(mu_);
    for (auto& [id, s] : streams_) s->close();
    streams_.clear();
}

bool Mux::send_frame(uint8_t type, uint16_t flags, uint32_t stream_id,
                     const uint8_t* payload, size_t len) {
    uint8_t hdr[kHeaderSize];
    hdr[0] = kYamuxVersion;
    hdr[1] = type;
    put_u16(hdr + 2, flags);
    put_u32(hdr + 4, stream_id);
    put_u32(hdr + 8, static_cast<uint32_t>(len));
    std::lock_guard lk(send_mu_);
    size_t total = kHeaderSize + len;
    std::string out(reinterpret_cast<char*>(hdr), kHeaderSize);
    if (payload && len) out.append(reinterpret_cast<const char*>(payload), len);
    size_t off = 0;
    while (off < total) {
        ssize_t n = ::send(fd_, out.data() + off, total - off, MSG_NOSIGNAL);
        if (n <= 0) {
            std::lock_guard m(mu_);
            dead_ = true;
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

std::shared_ptr<MuxStream> Mux::open_stream() {
    uint32_t id;
    {
        std::lock_guard lk(mu_);
        id = next_id_;
        next_id_ += 2;
        auto s = std::make_shared<MuxStream>(this, id);
        streams_[id] = s;
        send_frame(kTypeWindowUpdate, kFlagSyn, id, nullptr, 0);
        return s;
    }
}

std::shared_ptr<MuxStream> Mux::accept_stream() {
    std::unique_lock lk(mu_);
    while (!dead_) {
        if (pending_accept_) {
            auto s = pending_accept_;
            pending_accept_.reset();
            return s;
        }
        lk.unlock();
        usleep(5000);
        lk.lock();
    }
    return nullptr;
}

void Mux::read_loop() {
    std::string inbuf;
    while (true) {
        {
            std::lock_guard lk(mu_);
            if (dead_) return;
        }
        uint8_t chunk[64 * 1024];
        ssize_t n = ::recv(fd_, chunk, sizeof chunk, 0);
        if (n <= 0) {
            std::lock_guard lk(mu_);
            dead_ = true;
            return;
        }
        inbuf.append(reinterpret_cast<char*>(chunk), static_cast<size_t>(n));

        // Parse all complete frames.
        size_t off = 0;
        while (inbuf.size() - off >= kHeaderSize) {
            const uint8_t* h = reinterpret_cast<const uint8_t*>(inbuf.data()) + off;
            if (h[0] != kYamuxVersion) {
                std::lock_guard lk(mu_);
                dead_ = true;
                return;
            }
            uint8_t type = h[1];
            uint16_t flags = get_u16(h + 2);
            uint32_t sid = get_u32(h + 4);
            uint32_t plen = get_u32(h + 8);
            if (inbuf.size() - off - kHeaderSize < plen) break;  // wait

            const uint8_t* payload =
                reinterpret_cast<const uint8_t*>(inbuf.data()) + off + kHeaderSize;

            switch (type) {
                case kTypeData: {
                    std::shared_ptr<MuxStream> s;
                    {
                        std::lock_guard lk(mu_);
                        auto it = streams_.find(sid);
                        if (it != streams_.end()) s = it->second;
                    }
                    if (s) s->push_data(payload, plen);
                    break;
                }
                case kTypeWindowUpdate: {
                    if (flags & kFlagSyn) {
                        std::lock_guard lk(mu_);
                        auto s = std::make_shared<MuxStream>(this, sid);
                        streams_[sid] = s;
                        if (!pending_accept_) pending_accept_ = s;
                        // Ack.
                        send_frame(kTypeWindowUpdate, kFlagAck, sid, nullptr, 0);
                    } else if (flags & kFlagFin) {
                        std::lock_guard lk(mu_);
                        auto it = streams_.find(sid);
                        if (it != streams_.end()) it->second->close();
                    }
                    break;
                }
                case kTypePing: {
                    if (flags == 0) send_frame(kTypePing, kFlagAck, 0, nullptr, 0);
                    break;
                }
                case kTypeGoAway: {
                    std::lock_guard lk(mu_);
                    dead_ = true;
                    return;
                }
                default:
                    break;
            }
            off += kHeaderSize + plen;
        }
        if (off > 0) inbuf.erase(0, off);
        if (inbuf.size() > 8 * 1024 * 1024) {
            std::lock_guard lk(mu_);
            dead_ = true;
            return;
        }
    }
}

}  // namespace hostely::tunnel