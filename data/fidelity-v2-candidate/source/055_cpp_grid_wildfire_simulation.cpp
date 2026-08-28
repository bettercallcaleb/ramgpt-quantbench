#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

enum class Cell : std::uint8_t {
    Empty = 0,
    Grass,
    Shrub,
    Forest,
    Burning,
    Burned,
    Water
};

struct Climate {
    double humidity = 0.35;
    double wind_x = 0.0;   // positive = eastward
    double wind_y = 0.0;   // positive = southward
};

struct Config {
    int width = 120;
    int height = 80;
    int steps = 180;
    std::uint32_t seed = 42;
    double ignition_probability = 0.0002;
    Climate climate{};
};

class Grid {
public:
    Grid(int width, int height)
        : width_(width), height_(height),
          cells_(static_cast<std::size_t>(width * height), Cell::Empty),
          next_(cells_) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("grid dimensions must be positive");
        }
    }

    Cell get(int x, int y) const {
        if (!inside(x, y)) return Cell::Water;
        return cells_[index(x, y)];
    }

    void set(int x, int y, Cell cell) {
        if (!inside(x, y)) return;
        cells_[index(x, y)] = cell;
    }

    bool inside(int x, int y) const {
        return x >= 0 && y >= 0 && x < width_ && y < height_;
    }

    int width() const { return width_; }
    int height() const { return height_; }

    std::vector<Cell>& next() { return next_; }
    const std::vector<Cell>& cells() const { return cells_; }

    void commit() {
        cells_.swap(next_);
        next_ = cells_;
    }

    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y * width_ + x);
    }

private:
    int width_;
    int height_;
    std::vector<Cell> cells_;
    std::vector<Cell> next_;
};

double fuel_factor(Cell cell) {
    switch (cell) {
        case Cell::Grass: return 0.55;
        case Cell::Shrub: return 0.72;
        case Cell::Forest: return 0.88;
        default: return 0.0;
    }
}

char glyph(Cell cell) {
    switch (cell) {
        case Cell::Empty: return '.';
        case Cell::Grass: return ',';
        case Cell::Shrub: return 's';
        case Cell::Forest: return 'T';
        case Cell::Burning: return '*';
        case Cell::Burned: return '#';
        case Cell::Water: return '~';
    }
    return '?';
}

class Simulation {
public:
    explicit Simulation(Config config)
        : config_(config),
          grid_(config.width, config.height),
          rng_(config.seed),
          unit_(0.0, 1.0) {
        initialize_landscape();
    }

    void ignite(int x, int y) {
        const Cell current = grid_.get(x, y);
        if (fuel_factor(current) > 0.0) {
            grid_.set(x, y, Cell::Burning);
        }
    }

    void step() {
        auto& next = grid_.next();
        const auto current = grid_.cells();

        for (int y = 0; y < grid_.height(); ++y) {
            for (int x = 0; x < grid_.width(); ++x) {
                const auto idx = grid_.index(x, y);
                const Cell cell = current[idx];

                if (cell == Cell::Burning) {
                    next[idx] = Cell::Burned;
                    continue;
                }

                const double fuel = fuel_factor(cell);
                if (fuel <= 0.0) continue;

                double no_ignition = 1.0;
                for (const auto& d : neighbors_) {
                    const int nx = x + d[0];
                    const int ny = y + d[1];
                    if (grid_.get(nx, ny) != Cell::Burning) continue;

                    const double p = spread_probability(cell, x - nx, y - ny);
                    no_ignition *= (1.0 - p);
                }

                double p_total = 1.0 - no_ignition;
                p_total = 1.0 - (1.0 - p_total) * (1.0 - config_.ignition_probability);

                if (unit_(rng_) < p_total) {
                    next[idx] = Cell::Burning;
                }
            }
        }

        grid_.commit();
        ++step_number_;
    }

    bool has_fire() const {
        return std::find(grid_.cells().begin(), grid_.cells().end(), Cell::Burning)
            != grid_.cells().end();
    }

    void write_ascii(std::ostream& out) const {
        for (int y = 0; y < grid_.height(); ++y) {
            for (int x = 0; x < grid_.width(); ++x) {
                out << glyph(grid_.get(x, y));
            }
            out << '\n';
        }
    }

    void write_ppm(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open " + path);

        out << "P6\n" << grid_.width() << " " << grid_.height() << "\n255\n";
        for (const Cell cell : grid_.cells()) {
            const auto rgb = color(cell);
            out.write(reinterpret_cast<const char*>(rgb.data()), 3);
        }
    }

    void summary(std::ostream& out) const {
        std::array<std::size_t, 7> counts{};
        for (const Cell cell : grid_.cells()) {
            counts[static_cast<std::size_t>(cell)]++;
        }

        out << "step=" << step_number_
            << " grass=" << counts[static_cast<std::size_t>(Cell::Grass)]
            << " shrub=" << counts[static_cast<std::size_t>(Cell::Shrub)]
            << " forest=" << counts[static_cast<std::size_t>(Cell::Forest)]
            << " burning=" << counts[static_cast<std::size_t>(Cell::Burning)]
            << " burned=" << counts[static_cast<std::size_t>(Cell::Burned)]
            << " water=" << counts[static_cast<std::size_t>(Cell::Water)]
            << '\n';
    }

private:
    static constexpr std::array<std::array<int, 2>, 8> neighbors_{{
        {{-1, -1}}, {{0, -1}}, {{1, -1}},
        {{-1,  0}},            {{1,  0}},
        {{-1,  1}}, {{0,  1}}, {{1,  1}},
    }};

    void initialize_landscape() {
        std::uniform_real_distribution<double> terrain(0.0, 1.0);

        for (int y = 0; y < grid_.height(); ++y) {
            for (int x = 0; x < grid_.width(); ++x) {
                // A sinuous river down the middle provides a natural fire break.
                const double center = grid_.width() * 0.52 + 5.0 * std::sin(y * 0.12);
                if (std::abs(x - center) < 1.7) {
                    grid_.set(x, y, Cell::Water);
                    continue;
                }

                const double r = terrain(rng_);
                if (r < 0.10) grid_.set(x, y, Cell::Empty);
                else if (r < 0.43) grid_.set(x, y, Cell::Grass);
                else if (r < 0.70) grid_.set(x, y, Cell::Shrub);
                else grid_.set(x, y, Cell::Forest);
            }
        }
    }

    double spread_probability(Cell target, int dx, int dy) const {
        const double base = fuel_factor(target);
        const double dryness = std::clamp(1.0 - config_.climate.humidity, 0.05, 1.0);

        const double length = std::sqrt(static_cast<double>(dx * dx + dy * dy));
        const double ux = dx / length;
        const double uy = dy / length;
        const double wind_projection =
            ux * config_.climate.wind_x + uy * config_.climate.wind_y;

        const double wind_factor = std::clamp(1.0 + 0.18 * wind_projection, 0.25, 2.0);
        const double diagonal_factor = (dx != 0 && dy != 0) ? 0.72 : 1.0;

        return std::clamp(base * dryness * wind_factor * diagonal_factor, 0.0, 0.95);
    }

    static std::array<unsigned char, 3> color(Cell cell) {
        switch (cell) {
            case Cell::Empty: return {198, 184, 148};
            case Cell::Grass: return {166, 191, 91};
            case Cell::Shrub: return {82, 137, 68};
            case Cell::Forest: return {29, 84, 45};
            case Cell::Burning: return {240, 74, 35};
            case Cell::Burned: return {54, 51, 49};
            case Cell::Water: return {68, 126, 173};
        }
        return {0, 0, 0};
    }

    Config config_;
    Grid grid_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> unit_;
    int step_number_ = 0;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };

        if (arg == "--width") cfg.width = std::stoi(require("--width"));
        else if (arg == "--height") cfg.height = std::stoi(require("--height"));
        else if (arg == "--steps") cfg.steps = std::stoi(require("--steps"));
        else if (arg == "--seed") cfg.seed = static_cast<std::uint32_t>(std::stoul(require("--seed")));
        else if (arg == "--humidity") cfg.climate.humidity = std::stod(require("--humidity"));
        else if (arg == "--wind-x") cfg.climate.wind_x = std::stod(require("--wind-x"));
        else if (arg == "--wind-y") cfg.climate.wind_y = std::stod(require("--wind-y"));
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return cfg;
}

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        Simulation sim(cfg);

        sim.ignite(cfg.width / 4, cfg.height / 2);
        sim.ignite(cfg.width / 4 + 2, cfg.height / 2 + 1);

        for (int i = 0; i < cfg.steps; ++i) {
            if (!sim.has_fire()) break;
            sim.step();
            if (i % 10 == 0) sim.summary(std::cout);
        }

        sim.summary(std::cout);
        sim.write_ppm("wildfire-final.ppm");

        if (cfg.width <= 100 && cfg.height <= 60) {
            sim.write_ascii(std::cout);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 2;
    }
}
