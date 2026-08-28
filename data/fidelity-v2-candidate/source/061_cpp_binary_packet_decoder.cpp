#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Bounded decoder for a simple binary telemetry protocol.
//
// Frame:
//   0..1   magic        0xA5 0x5A
//   2      version      currently 1
//   3      flags
//   4..5   payload_len  big endian
//   6..9   sequence     big endian
//   10..13 unix_time    big endian
//   14..   payload
//   last4  crc32 over bytes [2, end-4)
//
// Payload TLV:
//   type:u8, length:u16 big endian, value:length bytes

class DecodeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Cursor {
public:
    explicit Cursor(std::span<const std::uint8_t> data) : data_(data) {}

    std::size_t remaining() const { return data_.size() - offset_; }
    std::size_t offset() const { return offset_; }

    std::uint8_t u8() {
        require(1);
        return data_[offset_++];
    }

    std::uint16_t u16be() {
        require(2);
        const std::uint16_t value =
            (static_cast<std::uint16_t>(data_[offset_]) << 8) |
            static_cast<std::uint16_t>(data_[offset_ + 1]);
        offset_ += 2;
        return value;
    }

    std::uint32_t u32be() {
        require(4);
        const std::uint32_t value =
            (static_cast<std::uint32_t>(data_[offset_]) << 24) |
            (static_cast<std::uint32_t>(data_[offset_ + 1]) << 16) |
            (static_cast<std::uint32_t>(data_[offset_ + 2]) << 8) |
            static_cast<std::uint32_t>(data_[offset_ + 3]);
        offset_ += 4;
        return value;
    }

    std::span<const std::uint8_t> bytes(std::size_t count) {
        require(count);
        auto result = data_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

private:
    void require(std::size_t count) const {
        if (count > remaining()) {
            throw DecodeError("truncated packet at offset " + std::to_string(offset_));
        }
    }

    std::span<const std::uint8_t> data_;
    std::size_t offset_ = 0;
};

std::uint32_t crc32(std::span<const std::uint8_t> data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

struct Tlv {
    std::uint8_t type{};
    std::vector<std::uint8_t> value;
};

struct Packet {
    std::uint8_t version{};
    std::uint8_t flags{};
    std::uint32_t sequence{};
    std::uint32_t unix_time{};
    std::vector<Tlv> fields;
};

std::vector<Tlv> decode_payload(std::span<const std::uint8_t> payload) {
    Cursor cur(payload);
    std::vector<Tlv> fields;

    constexpr std::size_t max_fields = 128;
    constexpr std::size_t max_single_value = 4096;

    while (cur.remaining() > 0) {
        if (fields.size() >= max_fields) {
            throw DecodeError("too many TLV fields");
        }

        const auto type = cur.u8();
        const auto length = cur.u16be();

        if (length > max_single_value) {
            throw DecodeError("TLV value exceeds configured maximum");
        }
        if (length > cur.remaining()) {
            throw DecodeError("TLV length exceeds remaining payload");
        }

        const auto data = cur.bytes(length);
        fields.push_back(Tlv{
            type,
            std::vector<std::uint8_t>(data.begin(), data.end())
        });
    }

    return fields;
}

Packet decode_packet(std::span<const std::uint8_t> frame) {
    constexpr std::size_t fixed_header = 14;
    constexpr std::size_t checksum_size = 4;
    constexpr std::size_t max_payload = 16384;

    if (frame.size() < fixed_header + checksum_size) {
        throw DecodeError("frame shorter than minimum packet size");
    }
    if (frame[0] != 0xA5 || frame[1] != 0x5A) {
        throw DecodeError("bad frame magic");
    }

    Cursor cur(frame.subspan(2));
    Packet packet;
    packet.version = cur.u8();
    packet.flags = cur.u8();

    if (packet.version != 1) {
        throw DecodeError("unsupported protocol version " + std::to_string(packet.version));
    }

    const auto payload_length = cur.u16be();
    if (payload_length > max_payload) {
        throw DecodeError("payload exceeds protocol maximum");
    }

    packet.sequence = cur.u32be();
    packet.unix_time = cur.u32be();

    const std::size_t expected_total = fixed_header + payload_length + checksum_size;
    if (frame.size() != expected_total) {
        std::ostringstream msg;
        msg << "length mismatch: header says payload=" << payload_length
            << ", frame bytes=" << frame.size()
            << ", expected=" << expected_total;
        throw DecodeError(msg.str());
    }

    const auto payload = cur.bytes(payload_length);
    const auto transmitted_crc = cur.u32be();

    const auto protected_region =
        frame.subspan(2, frame.size() - 2 - checksum_size);
    const auto calculated_crc = crc32(protected_region);

    if (calculated_crc != transmitted_crc) {
        std::ostringstream msg;
        msg << "CRC mismatch: expected 0x"
            << std::hex << std::setw(8) << std::setfill('0') << transmitted_crc
            << ", calculated 0x" << std::setw(8) << calculated_crc;
        throw DecodeError(msg.str());
    }

    packet.fields = decode_payload(payload);
    return packet;
}

std::string hex(std::span<const std::uint8_t> bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint8_t b : bytes) {
        out << std::setw(2) << static_cast<unsigned>(b);
    }
    return out.str();
}

std::optional<std::uint32_t> decode_u32_value(const Tlv& field) {
    if (field.value.size() != 4) return std::nullopt;
    return
        (static_cast<std::uint32_t>(field.value[0]) << 24) |
        (static_cast<std::uint32_t>(field.value[1]) << 16) |
        (static_cast<std::uint32_t>(field.value[2]) << 8) |
        static_cast<std::uint32_t>(field.value[3]);
}

void print_packet(const Packet& packet) {
    std::cout << "version=" << static_cast<unsigned>(packet.version)
              << " flags=0x" << std::hex << static_cast<unsigned>(packet.flags)
              << std::dec
              << " sequence=" << packet.sequence
              << " unix_time=" << packet.unix_time
              << " fields=" << packet.fields.size()
              << '\n';

    for (std::size_t i = 0; i < packet.fields.size(); ++i) {
        const auto& field = packet.fields[i];
        std::cout << "  [" << i << "] type="
                  << static_cast<unsigned>(field.type)
                  << " length=" << field.value.size();

        if (field.type == 1) {
            std::string text(field.value.begin(), field.value.end());
            std::cout << " text=" << std::quoted(text);
        } else if (field.type == 2) {
            if (auto number = decode_u32_value(field)) {
                std::cout << " u32=" << *number;
            } else {
                std::cout << " malformed-u32";
            }
        } else {
            std::cout << " hex=" << hex(field.value);
        }
        std::cout << '\n';
    }
}

std::vector<std::uint8_t> parse_hex_input(const std::string& text) {
    std::vector<std::uint8_t> bytes;
    int high = -1;

    for (char ch : text) {
        if (ch == ' ' || ch == '\n' || ch == '\t' || ch == ':') continue;

        int value = -1;
        if (ch >= '0' && ch <= '9') value = ch - '0';
        else if (ch >= 'a' && ch <= 'f') value = ch - 'a' + 10;
        else if (ch >= 'A' && ch <= 'F') value = ch - 'A' + 10;
        else throw DecodeError(std::string("invalid hex character: ") + ch);

        if (high < 0) {
            high = value;
        } else {
            bytes.push_back(static_cast<std::uint8_t>((high << 4) | value));
            high = -1;
        }
    }

    if (high >= 0) throw DecodeError("odd number of hex digits");
    return bytes;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: packet-decoder HEX_FRAME\n";
        return 2;
    }

    try {
        const auto bytes = parse_hex_input(argv[1]);
        const Packet packet = decode_packet(bytes);
        print_packet(packet);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "decode error: " << ex.what() << '\n';
        return 1;
    }
}
