#include "neuralforge/export_model.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace neuralforge {
namespace {

void write_values(std::ostream& out, const Tensor& t) {
    out << std::setprecision(9);
    for (std::size_t i = 0; i < t.size(); ++i) {
        out << t.at(i);
        if ((i + 1) % 8 == 0) out << '\n';
        else out << ' ';
    }
    out << '\n';
}

} // namespace

void export_mlp_model(const std::filesystem::path& path,
                      const Tensor& w1, const Tensor& b1,
                      const Tensor& w2, const Tensor& b2) {
    if (w1.ndim() != 2 || w2.ndim() != 2 || b1.ndim() != 1 || b2.ndim() != 1) {
        throw std::invalid_argument("export_mlp_model: unexpected tensor ranks");
    }
    if (w1.shape()[1] != b1.shape()[0] || w2.shape()[0] != b1.shape()[0] ||
        w2.shape()[1] != b2.shape()[0]) {
        throw std::invalid_argument("export_mlp_model: shape mismatch among weights");
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write model file: " + path.string());

    const auto n_in = w1.shape()[0];
    const auto n_hid = w1.shape()[1];
    const auto n_out = w2.shape()[1];

    out << "NEURALFORGE_MLP 1\n";
    out << n_in << ' ' << n_hid << ' ' << n_out << '\n';
    write_values(out, w1);
    write_values(out, b1);
    write_values(out, w2);
    write_values(out, b2);
}

} // namespace neuralforge
