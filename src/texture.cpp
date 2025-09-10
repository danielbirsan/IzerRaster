#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "texture.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring> // for std::memcpy
#include <iostream> // for std::cerr


Texture::Texture(const std::string& filename, bool uploadToGPU)
{
    int nChannels;
    uint8_t* data = stbi_load(filename.c_str(), &w, &h, &nChannels, 4);
    if (!data)
        throw std::runtime_error("stb_image failed for " + filename);

    pixels.resize(size_t(w) * h);
    std::memcpy(pixels.data(), data, pixels.size() * sizeof(uint32_t));
    stbi_image_free(data);

    if (uploadToGPU)
    {
        // ----- alocare pe device + copiere ------------
        size_t bytes = pixels.size() * sizeof(uint32_t);
        cudaMalloc(&device, bytes);
        cudaMemcpy(device, pixels.data(), bytes, cudaMemcpyHostToDevice);
    }
}

Texture::~Texture()
{
    if (device)
        cudaFree(device);
}
