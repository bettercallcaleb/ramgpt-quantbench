#include <atomic>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

// Single-producer / single-consumer lock-free ring buffer for interleaved audio.
// Capacity must be a power of two. The implementation leaves no sentinel slot;
// monotonic counters distinguish empty from full.

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert(Capacity >= 2, "Capacity must be at least two elements");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    bool push(const T& value) noexcept {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto tail = tail_.load(std::memory_order_acquire);
        if (head - tail >= Capacity) {
            return false;
        }
        storage_[index(head)] = value;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& value) noexcept {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto head = head_.load(std::memory_order_acquire);
        if (tail == head) {
            return false;
        }
        value = storage_[index(tail)];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    std::size_t size_approx() const noexcept {
        const auto head = head_.load(std::memory_order_acquire);
        const auto tail = tail_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(head - tail);
    }

    constexpr std::size_t capacity() const noexcept { return Capacity; }

    void reset_when_quiescent() noexcept {
        // Only call when producer and consumer are stopped.
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t mask_ = Capacity - 1;

    static constexpr std::size_t index(std::uint64_t counter) noexcept {
        return static_cast<std::size_t>(counter) & mask_;
    }

    alignas(64) std::array<T, Capacity> storage_{};
    alignas(64) std::atomic<std::uint64_t> head_{0};
    alignas(64) std::atomic<std::uint64_t> tail_{0};
};

struct StereoFrame {
    float left{};
    float right{};
};

class SineProducer {
public:
    explicit SineProducer(double sample_rate)
        : sample_rate_(sample_rate) {}

    StereoFrame next(double frequency_hz, double gain) {
        const double phase_inc = 2.0 * pi_ * frequency_hz / sample_rate_;
        phase_ += phase_inc;
        if (phase_ >= 2.0 * pi_) {
            phase_ -= 2.0 * pi_;
        }

        const float s = static_cast<float>(std::sin(phase_) * gain);
        return {s, s};
    }

private:
    static constexpr double pi_ = 3.14159265358979323846;
    double sample_rate_;
    double phase_{0.0};
};

struct Counters {
    std::atomic<std::uint64_t> produced{0};
    std::atomic<std::uint64_t> consumed{0};
    std::atomic<std::uint64_t> producer_overruns{0};
    std::atomic<std::uint64_t> consumer_underruns{0};
};

template <std::size_t Capacity>
void producer_thread(
    SpscRingBuffer<StereoFrame, Capacity>& queue,
    Counters& counters,
    std::atomic<bool>& running,
    double sample_rate
) {
    SineProducer osc(sample_rate);
    const auto frame_period = std::chrono::duration<double>(1.0 / sample_rate);
    auto deadline = std::chrono::steady_clock::now();

    while (running.load(std::memory_order_relaxed)) {
        deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_period);
        const auto frame = osc.next(440.0, 0.25);

        if (queue.push(frame)) {
            counters.produced.fetch_add(1, std::memory_order_relaxed);
        } else {
            counters.producer_overruns.fetch_add(1, std::memory_order_relaxed);
        }

        std::this_thread::sleep_until(deadline);
    }
}

template <std::size_t Capacity>
void consumer_thread(
    SpscRingBuffer<StereoFrame, Capacity>& queue,
    Counters& counters,
    std::atomic<bool>& running,
    std::vector<StereoFrame>& sink,
    double sample_rate
) {
    // Consume in blocks to resemble an audio callback.
    constexpr std::size_t block_size = 128;
    const auto block_period = std::chrono::duration<double>(block_size / sample_rate);
    auto deadline = std::chrono::steady_clock::now();

    while (running.load(std::memory_order_relaxed)) {
        deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(block_period);

        for (std::size_t i = 0; i < block_size; ++i) {
            StereoFrame frame{};
            if (queue.pop(frame)) {
                sink.push_back(frame);
                counters.consumed.fetch_add(1, std::memory_order_relaxed);
            } else {
                // A real callback would normally emit silence rather than block.
                sink.push_back({});
                counters.consumer_underruns.fetch_add(1, std::memory_order_relaxed);
            }
        }

        std::this_thread::sleep_until(deadline);
    }
}

void deterministic_unit_tests() {
    SpscRingBuffer<int, 8> q;
    assert(q.size_approx() == 0);

    for (int i = 0; i < 8; ++i) {
        assert(q.push(i));
    }
    assert(!q.push(9));
    assert(q.size_approx() == 8);

    for (int i = 0; i < 4; ++i) {
        int value = -1;
        assert(q.pop(value));
        assert(value == i);
    }
    assert(q.size_approx() == 4);

    for (int i = 8; i < 12; ++i) {
        assert(q.push(i));
    }

    for (int expected = 4; expected < 12; ++expected) {
        int value = -1;
        assert(q.pop(value));
        assert(value == expected);
    }

    int ignored = 0;
    assert(!q.pop(ignored));
    assert(q.size_approx() == 0);

    q.reset_when_quiescent();
    assert(q.push(42));
    assert(q.pop(ignored) && ignored == 42);
}

int main() {
    deterministic_unit_tests();

    constexpr std::size_t capacity = 4096;
    constexpr double sample_rate = 48000.0;

    SpscRingBuffer<StereoFrame, capacity> queue;
    Counters counters;
    std::atomic<bool> running{true};
    std::vector<StereoFrame> sink;
    sink.reserve(static_cast<std::size_t>(sample_rate * 3.0));

    std::thread producer(producer_thread<capacity>,
                         std::ref(queue),
                         std::ref(counters),
                         std::ref(running),
                         sample_rate);

    // Prefill slightly before starting consumer to reduce startup underruns.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    std::thread consumer(consumer_thread<capacity>,
                         std::ref(queue),
                         std::ref(counters),
                         std::ref(running),
                         std::ref(sink),
                         sample_rate);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    running.store(false, std::memory_order_relaxed);

    producer.join();
    consumer.join();

    double energy = 0.0;
    for (const auto& frame : sink) {
        energy += static_cast<double>(frame.left) * frame.left;
        energy += static_cast<double>(frame.right) * frame.right;
    }

    std::cout << "produced=" << counters.produced.load() << "\n";
    std::cout << "consumed=" << counters.consumed.load() << "\n";
    std::cout << "producer_overruns=" << counters.producer_overruns.load() << "\n";
    std::cout << "consumer_underruns=" << counters.consumer_underruns.load() << "\n";
    std::cout << "queue_remaining=" << queue.size_approx() << "\n";
    std::cout << "captured_frames=" << sink.size() << "\n";
    std::cout << "signal_energy=" << energy << "\n";

    return 0;
}
