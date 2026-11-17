/**
 * @file   color.cpp
 * @brief  Hashable color information for displaying semantic labels
 * @author Antoni Rosinol
 */

#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

#include "kimera_semantics_evidential/color.h"
#include "kimera_semantics_evidential/csv_iterator.h"

namespace kimera {

HashableColor::HashableColor(const Color& color) : Color(color) {}
HashableColor::HashableColor() : Color() {}
HashableColor::HashableColor(uint8_t r, uint8_t g, uint8_t b)
    : HashableColor(r, g, b, 255) {}
HashableColor::HashableColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    : Color(r, g, b, a) {}

bool HashableColor::operator==(const HashableColor& other) const {
  return (r == other.r && g == other.g && b == other.b && a == other.a);
}

bool HashableColor::equal(const HashableColor& color) const {
  return r == color.r && g == color.g && b == color.b && a == color.a;
}

size_t ColorHasher::operator()(const HashableColor& k) const {
  // Compute individual hash values for first,
  // second and third and combine them using XOR
  // and bit shifting:
  // TODO(Toni): use alpha value as well!!
  return ((std::hash<uint8_t>()(k.r) ^ (std::hash<uint8_t>()(k.g) << 1)) >> 1) ^
         (std::hash<uint8_t>()(k.b) << 1);
}

SemanticLabel2Color::SemanticLabel2Color(const std::string& filename)
    : color_to_semantic_label_(), semantic_label_to_color_map_() {
  std::ifstream file(filename.c_str());
  CHECK(file.good()) << "Couldn't open file: " << filename.c_str();
  size_t row_number = 1;
  for (CSVIterator loop(file); loop != CSVIterator(); ++loop) {
    // We expect the CSV to have header:
    // 0   , 1  , 2    , 3   , 4    , 5
    // name, red, green, blue, alpha, id
    CHECK_EQ(loop->size(), 6) << "Row " << row_number << " is invalid.";
    uint8_t r = std::atoi((*loop)[1].c_str());
    uint8_t g = std::atoi((*loop)[2].c_str());
    uint8_t b = std::atoi((*loop)[3].c_str());
    uint8_t a = std::atoi((*loop)[4].c_str());
    uint8_t id = std::atoi((*loop)[5].c_str());
    HashableColor rgba = HashableColor(r, g, b, a);
    semantic_label_to_color_map_[id] = rgba;
    color_to_semantic_label_[rgba] = id;
    row_number++;
  }
  // TODO(Toni): remove
  // Assign color 255,255,255 to unknown object 0u
  semantic_label_to_color_map_[kUnknownSemanticLabelId] =
      HashableColor::White();
  color_to_semantic_label_[HashableColor::White()] = kUnknownSemanticLabelId;
}

SemanticLabel SemanticLabel2Color::getSemanticLabelFromColor(
    const HashableColor& color) const {
  const auto& it = color_to_semantic_label_.find(color);
  if (it != color_to_semantic_label_.end()) {
    return it->second;
  } else {
    LOG(ERROR) << "Caught an unknown color: \n"
               << "RGBA: " << std::to_string(color.r) << ' '
               <<  std::to_string(color.g) << ' '
               <<  std::to_string(color.b) << ' '
               <<  std::to_string(color.a);
    return kUnknownSemanticLabelId; // Assign unknown label for now...
  }
}

HashableColor SemanticLabel2Color::getColorFromSemanticLabel(
    const SemanticLabel& semantic_label) const {
  const auto& it = semantic_label_to_color_map_.find(semantic_label);
  if (it != semantic_label_to_color_map_.end()) {
    HashableColor base_color = it-> second;
    return HashableColor(base_color.b, base_color.g,  base_color.r);
  } else {
    LOG(ERROR) << "Caught an unknown semantic label: \n"
               << "Label: " << std::to_string(semantic_label);
    return HashableColor(); // Assign unknown color for now...
  }
}

HashableColor SemanticLabel2Color::getColorFromSemanticState(const SemanticLabel& semantic_label, float value, ColorMode mode) {
  // Get base color for the label
  HashableColor base_color = getColorFromSemanticLabel(semantic_label);
  uint8_t adjusted_alpha = 255;
  if(mode == ColorMode::kSemantic) {
    // Use value as semantic label and keep alpha as is
    return HashableColor(base_color.b, base_color.g,  base_color.r, adjusted_alpha);
  } 
  else  if (mode == ColorMode::kSemanticProbability) {
    // Use value as probability and scale alpha relatively
    adjusted_alpha = static_cast<uint8_t>(255 * value);  
  } 
  else if (mode == ColorMode::kEvidential) {
    // Use value as evidence and clamp alpha to max 255
    float evid_strength = 3000.0f;
    value = std::max(0.0f,evid_strength);
    value = value /evid_strength;
    adjusted_alpha = static_cast<uint8_t>(255 * value);
  } 
  else {
    LOG(FATAL) << "Unknown color mode: "
               << static_cast<std::underlying_type<ColorMode>::type>(mode);
  }
 
  return HashableColor(base_color.b, base_color.g,  base_color.r, adjusted_alpha);
}

}  // namespace kimera