#pragma once
#include <cstdint>
#include <string>
#include <vector>

class Texture {
public:
    int                w = 0, h = 0;
    std::vector<uint32_t> pixels;   // ARGB8 pe CPU
    uint32_t*          device = nullptr;      // pointer în VRAM (cudaMalloc)

    explicit Texture(const std::string& filename, bool uploadToGPU = true);
    ~Texture();

    // copierea nu are sens (ar dubla cudaMalloc); o interzicem
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&)                 = default;
    Texture& operator=(Texture&&)      = default;
};
