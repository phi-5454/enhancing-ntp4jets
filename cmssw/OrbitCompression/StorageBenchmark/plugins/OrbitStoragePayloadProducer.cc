#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace {
  uint32_t readUint32(std::ifstream& input) {
    uint8_t bytes[4];
    input.read(reinterpret_cast<char*>(bytes), 4);
    if (!input)
      throw cms::Exception("OrbitStorage") << "Truncated uint32 in packed input";
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) |
           (uint32_t(bytes[3]) << 24);
  }

  unsigned headerInteger(const std::string& header, const std::string& name) {
    const std::regex pattern("\\\"" + name + "\\\":([0-9]+)");
    std::smatch match;
    if (!std::regex_search(header, match, pattern))
      throw cms::Exception("OrbitStorage") << "Missing " << name << " in packed header";
    return std::stoul(match[1].str());
  }

  class PackedEvents {
  public:
    explicit PackedEvents(const std::string& path) {
      std::ifstream input(path, std::ios::binary);
      if (!input)
        throw cms::Exception("OrbitStorage") << "Cannot open packed input " << path;
      char magic[8];
      input.read(magic, sizeof(magic));
      if (!input || std::string(magic, sizeof(magic)) != "ORBTPK01")
        throw cms::Exception("OrbitStorage") << "Invalid packed input magic in " << path;
      const auto headerSize = readUint32(input);
      std::string header(headerSize, '\0');
      input.read(header.data(), headerSize);
      if (!input)
        throw cms::Exception("OrbitStorage") << "Truncated packed header in " << path;
      bits_ = headerInteger(header, "bits_per_value");
      const auto eventCount = headerInteger(header, "event_count");
      if (bits_ == 0 || bits_ > 63)
        throw cms::Exception("OrbitStorage") << "Unsupported packed width " << bits_;
      const uint64_t mask = (uint64_t(1) << bits_) - 1;
      events_.reserve(eventCount);
      for (unsigned event = 0; event < eventCount; ++event) {
        const auto count = readUint32(input);
        const auto byteCount = (uint64_t(count) * bits_ + 7) / 8;
        std::vector<uint8_t> bytes(byteCount);
        input.read(reinterpret_cast<char*>(bytes.data()), byteCount);
        if (!input)
          throw cms::Exception("OrbitStorage") << "Truncated event " << event << " in " << path;
        std::vector<uint64_t> values;
        values.reserve(count);
        uint64_t accumulator = 0;
        unsigned accumulatorBits = 0;
        size_t offset = 0;
        for (unsigned index = 0; index < count; ++index) {
          while (accumulatorBits < bits_) {
            accumulator |= uint64_t(bytes.at(offset++)) << accumulatorBits;
            accumulatorBits += 8;
          }
          values.push_back(accumulator & mask);
          accumulator >>= bits_;
          accumulatorBits -= bits_;
        }
        events_.push_back(std::move(values));
      }
      if (input.peek() != std::char_traits<char>::eof())
        throw cms::Exception("OrbitStorage") << "Trailing data in packed input " << path;
    }

    unsigned bits() const { return bits_; }
    size_t size() const { return events_.size(); }
    const std::vector<uint64_t>& event(size_t index) const { return events_.at(index); }

  private:
    unsigned bits_ = 0;
    std::vector<std::vector<uint64_t>> events_;
  };

}  // namespace

class OrbitStoragePayloadProducer : public edm::one::EDProducer<> {
public:
  explicit OrbitStoragePayloadProducer(const edm::ParameterSet& config)
      : events_(config.getParameter<std::string>("inputFile")),
        representation_(config.getParameter<std::string>("representation")),
        outputKind_(config.getParameter<std::string>("outputKind")) {
    if (representation_ != "plain" && representation_ != "tokenized")
      throw cms::Exception("Configuration") << "representation must be plain or tokenized";
    if (outputKind_ != "edm" && outputKind_ != "nano")
      throw cms::Exception("Configuration") << "outputKind must be edm or nano";
    if (representation_ == "plain" && events_.bits() != 40)
      throw cms::Exception("Configuration") << "Plain PUPPI payload must use 40-bit words";

    if (outputKind_ == "nano") {
      produces<nanoaod::FlatTable>();
    } else if (representation_ == "plain") {
      produces<std::vector<uint16_t>>("pt");
      produces<std::vector<uint16_t>>("eta");
      produces<std::vector<uint16_t>>("phi");
      produces<std::vector<uint8_t>>("pid");
    } else if (events_.bits() <= 16) {
      produces<std::vector<uint16_t>>("token");
    } else {
      produces<std::vector<uint32_t>>("token");
    }
  }

  void produce(edm::Event& event, const edm::EventSetup&) override {
    if (eventIndex_ >= events_.size())
      throw cms::Exception("OrbitStorage") << "CMSSW requested more events than packed input contains";
    const auto& values = events_.event(eventIndex_++);
    if (outputKind_ == "nano")
      putNanoTable(event, values);
    else if (representation_ == "plain")
      putEdmPlainColumns(event, values);
    else
      putEdmTokens(event, values);
  }

private:
  void putEdmPlainColumns(edm::Event& event, const std::vector<uint64_t>& values) {
    auto pt = std::make_unique<std::vector<uint16_t>>();
    auto eta = std::make_unique<std::vector<uint16_t>>();
    auto phi = std::make_unique<std::vector<uint16_t>>();
    auto pid = std::make_unique<std::vector<uint8_t>>();
    pt->reserve(values.size());
    eta->reserve(values.size());
    phi->reserve(values.size());
    pid->reserve(values.size());
    for (auto word : values) {
      pt->push_back(word & 0x3FFF);
      eta->push_back((word >> 14) & 0x0FFF);
      phi->push_back((word >> 26) & 0x07FF);
      pid->push_back((word >> 37) & 7);
    }
    event.put(std::move(pt), "pt");
    event.put(std::move(eta), "eta");
    event.put(std::move(phi), "phi");
    event.put(std::move(pid), "pid");
  }

  template <typename T>
  void putEdmTokenVector(edm::Event& event, const std::vector<uint64_t>& values) {
    auto tokens = std::make_unique<std::vector<T>>();
    tokens->reserve(values.size());
    for (auto value : values)
      tokens->push_back(static_cast<T>(value));
    event.put(std::move(tokens), "token");
  }

  void putEdmTokens(edm::Event& event, const std::vector<uint64_t>& values) {
    if (events_.bits() <= 16)
      putEdmTokenVector<uint16_t>(event, values);
    else
      putEdmTokenVector<uint32_t>(event, values);
  }

  void putNanoTable(edm::Event& event, const std::vector<uint64_t>& values) {
    const std::string name = representation_ == "plain" ? "Puppi" : "Token";
    auto table = std::make_unique<nanoaod::FlatTable>(values.size(), name, false);
    if (representation_ == "plain") {
      std::vector<uint16_t> pt, eta, phi;
      std::vector<uint8_t> pid;
      pt.reserve(values.size());
      eta.reserve(values.size());
      phi.reserve(values.size());
      pid.reserve(values.size());
      for (auto word : values) {
        // Store the raw fixed-width codes in aligned integer leaves. Masking
        // (rather than sign extension) leaves all unused high bits at zero.
        pt.push_back(word & 0x3FFF);
        eta.push_back((word >> 14) & 0x0FFF);
        phi.push_back((word >> 26) & 0x07FF);
        pid.push_back((word >> 37) & 7);
      }
      table->addColumn<uint16_t>("pt", pt, "pT code; LSB 0.25 GeV");
      table->addColumn<uint16_t>("eta", eta, "12-bit signed eta code in low bits; LSB pi/720");
      table->addColumn<uint16_t>("phi", phi, "11-bit signed phi code in low bits; LSB pi/720");
      table->addColumn<uint8_t>("pid", pid, "3-bit PUPPI PID code in low bits");
    } else if (events_.bits() <= 16) {
      std::vector<uint16_t> tokens(values.begin(), values.end());
      table->addColumn<uint16_t>("value", tokens, "VQ token index");
    } else {
      std::vector<uint32_t> tokens(values.begin(), values.end());
      table->addColumn<uint32_t>("value", tokens, "VQ token index");
    }
    event.put(std::move(table));
  }

  PackedEvents events_;
  std::string representation_;
  std::string outputKind_;
  size_t eventIndex_ = 0;
};

DEFINE_FWK_MODULE(OrbitStoragePayloadProducer);
