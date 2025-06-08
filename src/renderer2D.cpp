#define SDL_MAIN_HANDLED
#include "renderer2D.hpp"
#include "render.h" // CUDA rasterizer declarations (initCuda, renderFrame, cleanupCuda)
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include "texture.hpp"
#include "render.h"        // pentru uploadTexture()


uint64_t Renderer2D::lastTime = 0;

Renderer2D::Renderer2D(const std::string appName, uint16_t width, uint16_t height) : appName(appName), windowWidth(width), windowHeight(height)
{
}

void Renderer2D::Init()
{
    SDL_SetAppMetadata(appName.c_str(), "1.0", "renderer");

    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus == cudaSuccess && deviceCount > 0) {
        std::cout << "CUDA device(s) found: " << deviceCount << ". Using GPU mode.\n";
        this->useGPU = true;
    } else {
        std::cout << "No CUDA devices found. Falling back to CPU mode.\n";
        this->useGPU = false;
    }

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }

    /*
    if (!TTF_Init())
    {
        std::cerr << "SDL_ttf could not initialize!"<< std::endl;
        SDL_Quit();
        return;
    }
    */
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");
    if (!SDL_CreateWindowAndRenderer(appName.c_str(), windowWidth, windowHeight, 0, &window, &renderer))
    {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        // TTF_Quit();
        SDL_Quit();

        return;
    }

    if (!SDL_SetRenderVSync(renderer, SDL_RENDERER_VSYNC_DISABLED)) {
    std::cerr << "SDL_SetRenderVSync failed: " << SDL_GetError() << "\n";
    }

    this->screenTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, windowWidth, windowHeight);
    if (!screenTexture)
    {
        std::cerr << "Screen texture could not be created! SDL Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        // TTF_Quit();
        SDL_Quit();

        return;
    }

    this->screenBuffer.resize(windowWidth * windowHeight);
    depthBufferCPU.resize(windowWidth * windowHeight, 1e9f); // iniţial infinit


    // font = TTF_OpenFont("arial.ttf", 24);

    // for triangle projections and geometry
    float fNear = 0.1f;
    float fFar = 1000.0f;
    float fFov = 60.0f;
    float fAspectRatio = (float)windowWidth / (float)windowHeight;

    proj = glm::perspective(glm::radians(fFov), fAspectRatio, fNear, fFar);

    cameraPos = glm::vec4{0};

    isRunning = true;

    this->mode = RenderMode::TEXTURED_WIREFRAME; // default render mode

    if (useGPU)
    {
        if (!initCuda(windowWidth, windowHeight))
        {
            std::cerr << "ERROR: CUDA rasterizer initialization failed!" << std::endl;
            // If GPU init fails, shut down and exit (no CPU fallback implemented here)
            Quit();
            return;
        }
        // Allocate host buffers for pixel color and depth (to receive CUDA output)
        cudaPixelBuffer = (uint32_t *)std::malloc(windowWidth * windowHeight * sizeof(uint32_t));
        cudaDepthBuffer = (float *)std::malloc(windowWidth * windowHeight * sizeof(float));

    }

     perfFreq = SDL_GetPerformanceFrequency();
    lastPerfCounter = SDL_GetPerformanceCounter();
}

uint64_t Renderer2D::GetCurrentTime()
{
    return SDL_GetTicks();
}

float Renderer2D::GetDeltaTime()
{
    return (GetCurrentTime() - Renderer2D::lastTime) / 1000.0f;
}

// While Running
void Renderer2D::Run()
{
    Renderer2D::lastTime = GetCurrentTime();
    while (isRunning)
    {
        float deltaTime = GetDeltaTime();
        Renderer2D::lastTime = GetCurrentTime();

        // Render one frame (input handled inside Render/UserUpdate)
        Render();
    }
}

void Renderer2D::Render()
{
    Uint64 frameStart = SDL_GetPerformanceCounter();

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    clearScreen();

    UserUpdate();

    SDL_UpdateTexture(screenTexture, nullptr, screenBuffer.data(), windowWidth * sizeof(RGBA));

    SDL_RenderTexture(renderer, screenTexture, nullptr, nullptr);

    SDL_RenderPresent(renderer);

      // end of frame
    Uint64 frameEnd = SDL_GetPerformanceCounter();
    double frameTime = double(frameEnd - frameStart) / double(perfFreq);  // seconds
    double fps = 1.0 / frameTime;

    // update window title with FPS every frame (or every N frames)
    char title[128];
    std::snprintf(title, sizeof(title), "%s — FPS: %.1f", appName.c_str(), fps);
    SDL_SetWindowTitle(window, title);
}

void Renderer2D::UserUpdate()
{
    for (auto &ev : poolInputEvents())
    {
        if (ev.type == "MOUSEWHEEL") {
            if (ev.wheelY > 0)      translate -= 1.0f;
            else if (ev.wheelY < 0) translate += 1.0f;
        }
        else if (ev.type == "KEYDOWN") {
            switch (ev.key) {
              case SDLK_ESCAPE: Quit();     break;
              case SDLK_W:      pitch_angle -= rotation_step; break;
              case SDLK_S:      pitch_angle += rotation_step; break;
              case SDLK_A:      theta       -= rotation_step; break;
              case SDLK_D:      theta       += rotation_step; break;
              case SDLK_R:
                if (!free_rotate) {
                  free_theta   = theta;
                  free_rotate  = true;
                } else {
                  theta       = free_theta;
                  pitch_angle = free_theta;
                  free_rotate = false;
                }
                break;
            }
        }
    }

    // 2) Continuous spin if R was toggled
    free_theta += GetDeltaTime();

    // 3) Build your model (object) transform exactly like your Python did:
    glm::mat4 T = glm::translate(glm::mat4(1.0f),
                                glm::vec3(0.0f, 0.0f, translate));
    glm::mat4 RX, RZ;
    if (!free_rotate) {
      RX = glm::rotate(glm::mat4(1.0f), pitch_angle, glm::vec3(1,0,0));
      RZ = glm::rotate(glm::mat4(1.0f), theta,       glm::vec3(0,0,1));
    } else {
      RX = glm::rotate(glm::mat4(1.0f), free_theta,  glm::vec3(1,0,0));
      RZ = glm::rotate(glm::mat4(1.0f), free_theta,  glm::vec3(0,0,1));
    }

    drawObj(applyRenderMatrix(T * RX * RZ, obj));
}

void Renderer2D::Quit()
{
    if (!isRunning && window == nullptr && renderer == nullptr)
    {
        return;
    }

    if (useGPU)
    {
        if (cudaPixelBuffer)
        {
            std::free(cudaPixelBuffer);
            cudaPixelBuffer = nullptr;
        }
        if (cudaDepthBuffer)
        {
            std::free(cudaDepthBuffer);
            cudaDepthBuffer = nullptr;
        }
        cleanupCuda();
    }

    if (renderer)
    {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    if (window)
    {
        SDL_DestroyWindow(window);
        window = nullptr;
    }

    // TTF_CloseFont(font);
    // TTF_Quit();

    SDL_Quit();

    isRunning = false;
}

void Renderer2D::clearScreen()
{
    std::fill(screenBuffer.begin(), screenBuffer.end(), RGBA());
    std::fill(depthBufferCPU.begin(), depthBufferCPU.end(), 1e9f);
}


void Renderer2D::drawPoint(uint16_t x, uint16_t y, RGBA rgba)
{
    if (y < 0 || y >= windowHeight || x < 0 || x >= windowWidth)
        return;

    this->screenBuffer[y * windowWidth + x] = rgba;
}

void Renderer2D::drawLine(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba) // Brensenham's Line Algorithm
{
    int dx = std::abs(x2 - x1);
    int dy = -std::abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;

    int err = dx + dy;

    int currentX = x1;
    int currentY = y1;

    while (true)
    {
        drawPoint(currentX, currentY, rgba);

        if (currentX == x2 && currentY == y2)
            break;

        if (2 * err >= dy)
        {
            if (currentX == x2)
                break;
            err += dy;
            currentX += sx;
        }

        if (2 * err <= dx)
        {
            if (currentY == y2)
                break;
            err += dx;
            currentY += sy;
        }
    }
}

void Renderer2D::drawRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba)
{
    drawLine(x1, y1, x2, y1, rgba);
    drawLine(x1, y1, x1, y2, rgba);
    drawLine(x2, y2, x1, y2, rgba);
    drawLine(x2, y2, x2, y1, rgba);
}

void Renderer2D::fillRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba)
{
    int startX = std::min(x1, x2);
    int startY = std::min(y1, y2);
    int endX = std::max(x1, x2);
    int endY = std::max(y1, y2);

    for (uint16_t y = startY; y < endY; y++)
    {
        drawHorizontalLine(y, startX, endX, rgba);
    }
}

// https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
void Renderer2D::drawCircle(uint16_t x, uint16_t y, uint16_t radius, RGBA rgba)
{
    if (radius < 0)
        return;

    if (radius == 0)
    {
        drawPoint(x, y, rgba);
        return;
    }

    // midpoint circle algorithm
    int startX = 0;
    int startY = radius;

    int decision = 3 - 2 * radius;

    auto plotOctants = [&](int offsetX, int offsetY) // so you divide the space into 8 parts, and draw in each of them
    {
        drawPoint(x + offsetX, y + offsetY, rgba);
        drawPoint(x - offsetX, y + offsetY, rgba);
        drawPoint(x + offsetX, y - offsetY, rgba);
        drawPoint(x - offsetX, y - offsetY, rgba);
        drawPoint(x + offsetY, y + offsetX, rgba);
        drawPoint(x - offsetY, y + offsetX, rgba);
        drawPoint(x + offsetY, y - offsetX, rgba);
        drawPoint(x - offsetY, y - offsetX, rgba);
    };

    plotOctants(startX, startY); // point first points on the axes

    while (startX <= startY)
    {
        startX++; // you always move horizontally

        if (decision > 0) // if true, you also need to move vertically
        {
            // midpoints was outside the circle
            startY--; // y-- because the upper left corner is (0,0), so going up means, decrementing

            decision = decision + 4 * (startX - startY) + 10;
        }
        else // midpoint was on the circle or inside it
            decision = decision + 4 * startX + 6;

        plotOctants(startX, startY);
    }
}

void Renderer2D::drawHorizontalLine(uint16_t y, uint16_t x1, uint16_t x2, RGBA rgba)
{
    if (x1 > x2)
        std::swap(x1, x2);

    if (y < 0 || y >= windowHeight)
        return;

    uint16_t startX = x1;
    uint16_t endX = std::min(windowWidth, x2 + 1);

    if (startX >= endX)
        return;

    auto rowStart = screenBuffer.begin() + y * windowWidth + startX;
    std::fill(rowStart, rowStart + (endX - startX), rgba);
}

void Renderer2D::fillCircle(uint16_t x, uint16_t y, uint16_t radius, RGBA rgba)
{
    if (radius < 0)
        return;

    if (radius == 0)
    {
        drawPoint(x, y, rgba);
        return;
    }

    // midpoint circle algorithm
    uint16_t startX = 0;
    uint16_t startY = radius;

    int decision = 3 - 2 * radius;

    auto plotHorizontalLines = [&](uint16_t offsetX, uint16_t offsetY) // this is the only thing that changes from the drawCircle function
    {
        drawHorizontalLine(y + offsetY, x - offsetX, x + offsetX, rgba); // bottom

        drawHorizontalLine(y - offsetY, x - offsetX, x + offsetX, rgba); // top

        drawHorizontalLine(y + offsetX, x - offsetY, x + offsetY, rgba); // mid bottom

        drawHorizontalLine(y - offsetX, x - offsetY, x + offsetY, rgba); // mid top
    };

    plotHorizontalLines(startX, startY); // point first points on the axes

    while (startX <= startY)
    {
        startX++; // you always move horizontally

        if (decision > 0) // if true, you also need to move vertically
        {
            // midpoints was outside the circle
            startY--; // y-- because the upper left corner is (0,0), so going up means, decrementing

            decision = decision + 4 * (startX - startY) + 10;
        }
        else // midpoint was on the circle or inside it
            decision = decision + 4 * startX + 6;

        plotHorizontalLines(startX, startY);
    }
}

void Renderer2D::drawTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba)
{
    drawLine(x1, y1, x2, y2, rgba);
    drawLine(x2, y2, x3, y3, rgba);
    drawLine(x1, y1, x3, y3, rgba);
}

void Renderer2D::fillBottomFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba)
{
    float slope1 = (float)(x2 - x1) / (y2 - y1);
    float slope2 = (float)(x3 - x1) / (y3 - y1);

    float currentX1 = x1;
    float currentX2 = x1;

    for (uint16_t startY = y1; startY <= y2; startY++)
    {
        drawHorizontalLine(startY, (uint16_t)currentX1, (uint16_t)currentX2, rgba);
        currentX1 += slope1;
        currentX2 += slope2;
    }
}

void Renderer2D::fillTopFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba)
{
    float slope1 = (float)(x2 - x1) / (y2 - y1);
    float slope2 = (float)(x3 - x1) / (y3 - y1);

    float currentX1 = x1;
    float currentX2 = x1;

    for (uint16_t startY = y1; startY >= y2; startY--)
    {
        drawHorizontalLine(startY, (uint16_t)currentX1, (uint16_t)currentX2, rgba);
        currentX1 -= slope1;
        currentX2 -= slope2;
    }
}

// https://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
// https://gamedev.stackexchange.com/questions/178181/how-can-i-draw-filled-triangles-in-c
// https://www.gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
void Renderer2D::fillTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba) // scanline triangle rasterization algorithm
{
    std::vector<std::pair<uint16_t, uint16_t>> vertices;
    vertices.push_back(std::make_pair(x1, y1));
    vertices.push_back(std::make_pair(x2, y2));
    vertices.push_back(std::make_pair(x3, y3));

    std::sort(vertices.begin(), vertices.end(), [](const std::pair<uint16_t, uint16_t> a, const std::pair<uint16_t, uint16_t> b)
              { return (a.second < b.second) || (a.second == b.second && a.first < b.first); });

    std::pair<uint16_t, uint16_t> v0 = vertices[0];
    std::pair<uint16_t, uint16_t> v1 = vertices[1];
    std::pair<uint16_t, uint16_t> v2 = vertices[2];

    // so you basically divide the triangle further into two triangles
    // one with a flat bottom and one with a flat top
    // if it does not already have a flat top or bottom

    if ((v1.first - v0.first) * (v2.second - v0.second) == (v2.first - v0.first) * (v1.second - v0.second))
        return;

    if (v0.second == v1.second && v1.second == v2.second)
        return;

    if (v1.second == v2.second)
    {
        fillBottomFlatTriangle(v0.first, v0.second, v1.first, v1.second, v2.first, v2.second, rgba);
    }
    else if (v0.second == v1.second)
    {
        fillTopFlatTriangle(v2.first, v2.second, v1.first, v1.second, v0.first, v0.second, rgba);
    }
    else
    {
        // create a new vertex
        std::pair<uint16_t, uint16_t> v3 = std::make_pair(v0.first + (uint16_t)((float)(v1.second - v0.second) / (v2.second - v0.second) * (v2.first - v0.first)), v1.second);
        fillBottomFlatTriangle(v0.first, v0.second, v1.first, v1.second, v3.first, v3.second, rgba);
        fillTopFlatTriangle(v2.first, v2.second, v3.first, v3.second, v1.first, v1.second, rgba);
    }
}

// Matrix Render for each triangle handle
mesh Renderer2D::applyRenderMatrix(glm::mat4 mat, mesh objMesh)
{
    mesh newMesh;

    if (useGPU)
    {

        currentTransform = mat;
        return objMesh;
    }
    for (const auto& tri : objMesh.tris) {
    triangle nt;
    nt.p[0] = mat * tri.p[0];
    nt.p[1] = mat * tri.p[1];
    nt.p[2] = mat * tri.p[2];

    // ► COPIEM şi UV-urile!
    nt.t[0] = tri.t[0];
    nt.t[1] = tri.t[1];
    nt.t[2] = tri.t[2];

    newMesh.tris.push_back(nt);
}


    return newMesh;
}



// .obj load from path
mesh Renderer2D::loadObj(std::string path)
{
    bool ok = obj.LoadFromObjectFile(path);
    currentMesh = &obj;

    // 1) Verificăm dacă toate UV-urile sunt (0,0). 
    //    Dacă da → înseamnă că fişierul OBJ nu conţine vt şi trebuie să generăm noi UV.
    bool needGenerateUV = true;
    for (auto &tri : obj.tris) {
        for (int i = 0; i < 3; ++i) {
            if (tri.t[i] != glm::vec2(0.0f, 0.0f)) {
                needGenerateUV = false;
                break;
            }
        }
        if (!needGenerateUV) break;
    }

    // 2) Generare UV sferic dacă e cazul
    if (needGenerateUV) {
        for (auto &tri : obj.tris) {
            for (int i = 0; i < 3; ++i) {
                // Normalizăm punctul pentru a-l aduce pe sfera unitate:
                glm::vec3 p = glm::normalize(glm::vec3(tri.p[i]));
                // Calculăm coordonatele UV sferice:
                //   – u variaza de la 0 la 1 → pe orizontală folosind atan2
                //   – v variaza de la 0 la 1 → pe verticală folosind asin
                float u = 0.5f + (atan2f(p.z, p.x) / (2.0f * glm::pi<float>()));
                float v = 0.5f - (asinf(p.y)          / glm::pi<float>());
                tri.t[i] = glm::vec2(u, v);
            }
        }
    }

    std::cout << "Loaded triangles: " << obj.tris.size()
              << (needGenerateUV ? " (UV generated spherically)\n"
                                 : " (UV loaded from file)\n");

    return obj;
}



// .obj drawing
void Renderer2D::drawObj(mesh obj)
{
    currentMesh = const_cast<mesh*>(&obj); // set the current mesh to the one being drawn
    if (useGPU)
    {
        gpuRender(obj);
    }
    else
    {
        simpleRender(obj);
    }
}

// mesh randering
void Renderer2D::simpleRender(mesh meshObj)
{
    std::vector<triangle> tria;

    triangle triProjected;

    // glm::mat4 transl = glm::translate(glm::vec3(0.0f,0.0f,8.0f));

    // mesh newMesh = applyRenderMatrix(transl * rotX * rotZ, meshObj);

    for (auto triTranslated : meshObj.tris)
    {

        // Calculating the normals in order to display the visible triangles
        glm::vec3 normal, line1, line2;
        line1.x = triTranslated.p[1].x - triTranslated.p[0].x;
        line1.y = triTranslated.p[1].y - triTranslated.p[0].y;
        line1.z = triTranslated.p[1].z - triTranslated.p[0].z;

        line2.x = triTranslated.p[2].x - triTranslated.p[0].x;
        line2.y = triTranslated.p[2].y - triTranslated.p[0].y;
        line2.z = triTranslated.p[2].z - triTranslated.p[0].z;

        normal.x = line1.y * line2.z - line1.z * line2.y;
        normal.y = line1.z * line2.x - line1.x * line2.z;
        normal.z = line1.x * line2.y - line1.y * line2.x;

        float l = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        normal.x /= l;
        normal.y /= l;
        normal.z /= l;

        // Display the visible triangles
        if (normal.x * (triTranslated.p[0].x - cameraPos.x) + normal.y * (triTranslated.p[0].y - cameraPos.y) + normal.z * (triTranslated.p[0].z - cameraPos.z) > 0.0f)
        {

            for (int i = 0; i < 3; ++i) {
    triProjected.p[i] = proj * triTranslated.p[i];
    triProjected.t[i] = triTranslated.t[i];   //  ▲▲▲  păstrează UV-ul
}


            for (int i = 0; i < 3; i++)
            {
                triProjected.p[i].x /= triProjected.p[i].w;
                triProjected.p[i].y /= triProjected.p[i].w;
                triProjected.p[i].z /= triProjected.p[i].w;
                triProjected.p[i].w  = 1.0f;  
            }

            // Scale
            triProjected.p[0].x += 1.0f;
            triProjected.p[0].y += 1.0f;
            triProjected.p[1].x += 1.0f;
            triProjected.p[1].y += 1.0f;
            triProjected.p[2].x += 1.0f;
            triProjected.p[2].y += 1.0f;
            triProjected.p[0].x *= 0.5f * (float)windowWidth;
            triProjected.p[0].y *= 0.5f * (float)windowHeight;
            triProjected.p[1].x *= 0.5f * (float)windowWidth;
            triProjected.p[1].y *= 0.5f * (float)windowHeight;
            triProjected.p[2].x *= 0.5f * (float)windowWidth;
            triProjected.p[2].y *= 0.5f * (float)windowHeight;

            tria.push_back(triProjected);
        }
    }

    // Sorting triagles for correct drawing order
    sort(tria.begin(), tria.end(), [](triangle &t1, triangle &t2)
         {
			float z1 = (t1.p[0].z + t1.p[1].z + t1.p[2].z) / 3.0f;
			float z2 = (t2.p[0].z + t2.p[1].z + t2.p[2].z) / 3.0f;
			return z1 > z2; });

    // Drawing
    for (const auto &triToRaster : tria)
    {
        if (this->mode == RenderMode::SHADED || this->mode == RenderMode::SHADED_WIREFRAME)
            fillTriangle(
                static_cast<uint16_t>(triToRaster.p[0].x), static_cast<uint16_t>(triToRaster.p[0].y),
                static_cast<uint16_t>(triToRaster.p[1].x), static_cast<uint16_t>(triToRaster.p[1].y),
                static_cast<uint16_t>(triToRaster.p[2].x), static_cast<uint16_t>(triToRaster.p[2].y),
                RGBA(200, 200, 200, 255));

        if (this->mode == RenderMode::WIREFRAME || this->mode == RenderMode::SHADED_WIREFRAME)
            drawTriangle(
                static_cast<uint16_t>(triToRaster.p[0].x), static_cast<uint16_t>(triToRaster.p[0].y),
                static_cast<uint16_t>(triToRaster.p[1].x), static_cast<uint16_t>(triToRaster.p[1].y),
                static_cast<uint16_t>(triToRaster.p[2].x), static_cast<uint16_t>(triToRaster.p[2].y),
                RGBA(100, 100, 100, 200));
   
        if (mode == RenderMode::TEXTURED || mode == RenderMode::TEXTURED_WIREFRAME)
            {
                fillTexturedTri(triToRaster, currentTex ? currentTex : meshObj.texture);
            }
         if (mode == RenderMode::SHADED_WIREFRAME || mode == RenderMode::WIREFRAME || mode == RenderMode::TEXTURED_WIREFRAME)
        {
            // Draw wireframe
            drawLine(static_cast<uint16_t>(triToRaster.p[0].x), static_cast<uint16_t>(triToRaster.p[0].y),
                     static_cast<uint16_t>(triToRaster.p[1].x), static_cast<uint16_t>(triToRaster.p[1].y), RGBA(255, 255, 255, 255));
            drawLine(static_cast<uint16_t>(triToRaster.p[1].x), static_cast<uint16_t>(triToRaster.p[1].y),
                     static_cast<uint16_t>(triToRaster.p[2].x), static_cast<uint16_t>(triToRaster.p[2].y), RGBA(255, 255, 255, 255));
            drawLine(static_cast<uint16_t>(triToRaster.p[2].x), static_cast<uint16_t>(triToRaster.p[2].y),
                     static_cast<uint16_t>(triToRaster.p[0].x), static_cast<uint16_t>(triToRaster.p[0].y), RGBA(255, 255, 255, 255));
        }

   }
}

void Renderer2D::drawCube()
{
    meshCube.tris = {
        {glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)},
        {glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)},

        {glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)},
        {glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)},

        {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f)},
        {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)},

        {glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)},
        {glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)},

        {glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)},
        {glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)},

        {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)},
        {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)},

    };
    simpleRender(meshCube);
}



void Renderer2D::gpuRender(const mesh &newObj)
{
    if (newObj.tris.empty())
    {
        // Dacă nu avem niciun mesh, ieşim imediat
        return;
    }

    // Timp (nu e folosit acum, dar rămâne pentru eventuale animații)
    float time = SDL_GetTicks() * 0.001f;

    // 1) Construim matricile View / Projection
    glm::mat4 viewMat = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -5));
    glm::mat4 projMat = glm::perspective(
        glm::radians(60.0f),
        float(windowWidth) / float(windowHeight),
        0.1f, 100.0f
    );

    // ModelViewProjection combinat
    glm::mat4 mvpMat = projMat * viewMat * currentTransform;
    // ModelView pentru culling (dacă vrem)
    glm::mat4 mvMat = viewMat * currentTransform;

    // 2) Transformăm fiecare triunghi în spațiul ecran (cu UV)
    std::vector<Tri> triList;
    triList.reserve(newObj.tris.size());

    for (auto &tri : newObj.tris)
    {
        // ── Back-face culling dacă vrei, acum e comentat
        // glm::vec4 v0 = mvMat * tri.p[0];
        // glm::vec4 v1 = mvMat * tri.p[1];
        // glm::vec4 v2 = mvMat * tri.p[2];
        // glm::vec3 e1 = glm::vec3(v1 - v0);
        // glm::vec3 e2 = glm::vec3(v2 - v0);
        // glm::vec3 normal = glm::normalize(glm::cross(e1, e2));
        // if (glm::dot(normal, glm::vec3(v0)) <= 0.0f) continue;

        // 2.1) Aplicăm proiecția MVP pe vârfuri
        glm::vec4 p0 = mvpMat * tri.p[0];
        glm::vec4 p1 = mvpMat * tri.p[1];
        glm::vec4 p2 = mvpMat * tri.p[2];

        // 2.2) Perspective divide → NDC
        p0 /= p0.w;
        p1 /= p1.w;
        p2 /= p2.w;

        // 2.3) Din coordonate NDC în coordonate ecran (pixeli)
        auto toScreen = [&](const glm::vec4 &v)
        {
            return glm::vec2(
                (v.x * 0.5f + 0.5f) * windowWidth,
                (1.0f - (v.y * 0.5f + 0.5f)) * windowHeight
            );
        };

        Tri projTri;
        // Poziții ecran
        projTri.p[0] = toScreen(p0);
        projTri.p[1] = toScreen(p1);
        projTri.p[2] = toScreen(p2);

        // Adâncimi
        projTri.z[0] = p0.z;
        projTri.z[1] = p1.z;
        projTri.z[2] = p2.z;

        // COPIEM UV-urile
        projTri.t[0] = tri.t[0];
        projTri.t[1] = tri.t[1];
        projTri.t[2] = tri.t[2];

        // Media adâncimilor (pentru sortare)
        projTri.depth = (p0.z + p1.z + p2.z) / 3.0f;

        // Adăugăm tri la listă
        triList.push_back(projTri);
    }

    // 3) Sortăm triunghiurile după adâncime (far → near)
    std::sort(triList.begin(), triList.end(), [](const Tri &a, const Tri &b)
    {
        return a.depth > b.depth;
    });

    // 4) Pregătim buffer-ul de CudaTri pentru CUDA
    std::vector<CudaTri> cudaTris;
    cudaTris.reserve(triList.size());

    for (const Tri &triCPU : triList)
    {
        CudaTri ct{};
        // Coordonate ecran (int)
        ct.x0 = triCPU.p[0].x;  ct.y0 = triCPU.p[0].y;
        ct.x1 = triCPU.p[1].x;  ct.y1 = triCPU.p[1].y;
        ct.x2 = triCPU.p[2].x;  ct.y2 = triCPU.p[2].y;

        // Adâncimi
        ct.z0 = triCPU.z[0];
        ct.z1 = triCPU.z[1];
        ct.z2 = triCPU.z[2];

        // UV-urile (float)
        ct.u0 = std::fmod(triCPU.t[0].x, 1.0f);  if (ct.u0 < 0) ct.u0 += 1.0f;
        ct.v0 = std::fmod(triCPU.t[0].y, 1.0f);  if (ct.v0 < 0) ct.v0 += 1.0f;
/* idem u1/v1 şi u2/v2 */
        ct.u1 = std::fmod(triCPU.t[1].x, 1.0f);  if (ct.u1 < 0) ct.u1 += 1.0f;
        ct.v1 = std::fmod(triCPU.t[1].y, 1.0f);  if (ct.v1 < 0) ct.v1 += 1.0f;
      ct.u2 = std::fmod(triCPU.t[2].x, 1.0f);  if (ct.u2 < 0) ct.u2 += 1.0f;
        ct.v2 = std::fmod(triCPU.t[2].y, 1.0f);  if (ct.v2 < 0) ct.v2 += 1.0f;

        cudaTris.push_back(ct);
    }

    // 5) Apelăm kernel-ul CUDA care umple cudaPixelBuffer + cudaDepthBuffer
    renderFrame(
        cudaTris.data(),
        static_cast<int>(cudaTris.size()),
        cudaPixelBuffer,
        cudaDepthBuffer
    );

    // 6) Dacă e modul WIREFRAME pur, punem fundal negru
    if (mode == RenderMode::WIREFRAME)
    {
        std::fill_n(cudaPixelBuffer, windowWidth * windowHeight, 0xFF000000u);
    }

    // 7) Dacă e oricare overlay „-WIREFRAME”, desenăm contururile peste buffer
    if (mode == RenderMode::WIREFRAME ||
        mode == RenderMode::SHADED_WIREFRAME ||
        mode == RenderMode::TEXTURED_WIREFRAME)
    {
        uint32_t lineColor = 0xFF00FF00u;
        auto clampPoint = [&](int &x, int &y)
        {
            return x >= 0 && x < windowWidth && y >= 0 && y < windowHeight;
        };
        auto drawLine = [&](int x0, int y0, float z0, int x1, int y1, float z1)
        {
            int dx = std::abs(x1 - x0), sx = (x0 < x1 ? 1 : -1);
            int dy = -std::abs(y1 - y0), sy = (y0 < y1 ? 1 : -1);
            int err = dx + dy;
            int steps = std::max(dx, -dy);
            for (int i = 0; i <= steps; ++i)
            {
                float t = (steps > 0 ? float(i) / steps : 0.0f);
                float z = z0 * (1 - t) + z1 * t;
                if (clampPoint(x0, y0))
                {
                    int idx = y0 * windowWidth + x0;
                    if (z <= cudaDepthBuffer[idx] + 1e-5f)
                    {
                        cudaPixelBuffer[idx] = lineColor;
                    }
                }
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        };

        // Pentru fiecare triunghi din triList (aceeași ordine ca mai sus)
        for (auto &T : triList)
        {
            int x0 = int(T.p[0].x + 0.5f), y0 = int(T.p[0].y + 0.5f);
            int x1 = int(T.p[1].x + 0.5f), y1 = int(T.p[1].y + 0.5f);
            int x2 = int(T.p[2].x + 0.5f), y2 = int(T.p[2].y + 0.5f);
            drawLine(x0, y0, T.z[0], x1, y1, T.z[1]);
            drawLine(x1, y1, T.z[1], x2, y2, T.z[2]);
            drawLine(x2, y2, T.z[2], x0, y0, T.z[0]);
        }
    }

    // 8) Copiem în final din buffer-ul CUDA în screenBuffer (format RGBA32)
    std::memcpy(
        screenBuffer.data(),
        cudaPixelBuffer,
        windowWidth * windowHeight * sizeof(uint32_t)
    );

    // Un singur pixel roșu în centru, ca „semn de viețuire”
    int cx = windowWidth / 2, cy = windowHeight / 2;
    if (cx >= 0 && cx < windowWidth && cy >= 0 && cy < windowHeight)
    {
        screenBuffer[cy * windowWidth + cx] = RGBA(255, 0, 0, 255);
    }
}

// OBJ loading cu suport complet pentru UV-uri (vt) şi, opcional, normale (vn)
bool mesh::LoadFromObjectFile(const std::string& sFilename)
{
    std::ifstream f(sFilename);
    if (!f.is_open())
    {
        std::cerr << "ERROR: LoadFromObjectFile - Cannot open file: " << sFilename << '\n';
        return false;
    }

    /* --- Buffere temporare pentru datele brute din OBJ ------------------- */
    std::vector<glm::vec4> temp_verts;      // v
    std::vector<glm::vec2> temp_texCoords;  // vt
    std::vector<glm::vec3> temp_normals;    // vn

    tris.clear();                           // ieşirea finală

    /* --- Variabile utilitare --------------------------------------------- */
    std::string line;
    uint32_t line_number = 0;
    bool load_error      = false;

    /* --- Struct intern pentru indici v / vt / vn per-vertex -------------- */
    struct IndexTriplet {
        int v  = 0;   // index poziţie (obligatoriu)
        int vt = 0;   // index UV        (opţional)
        int vn = 0;   // index normal    (opţional)
    };

    while (std::getline(f, line))
    {
        ++line_number;
        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command.empty() || command[0] == '#')
            continue;

        /* -------------------- VERTEX POSITION ----------------------------- */
        if (command == "v")
        {
            glm::vec4 v;
            if (!(ss >> v.x >> v.y >> v.z))
            {
                std::cerr << "WARNING: Line " << line_number
                          << ": Malformed vertex data.\n";
                load_error = true;
                continue;
            }
            v.w = 1.0f;
            temp_verts.push_back(v);
        }
        /* -------------------- VERTEX TEXCOORD ----------------------------- */
        else if (command == "vt")
        {
            glm::vec2 vt;
            if (!(ss >> vt.x >> vt.y))
            {
                std::cerr << "WARNING: Line " << line_number
                          << ": Malformed texture coordinate data.\n";
                load_error = true;
                continue;
            }
            temp_texCoords.push_back(vt);
        }
        /* -------------------- VERTEX NORMAL ------------------------------- */
        else if (command == "vn")
        {
            glm::vec3 vn;
            if (!(ss >> vn.x >> vn.y >> vn.z))
            {
                std::cerr << "WARNING: Line " << line_number
                          << ": Malformed normal data.\n";
                load_error = true;
                continue;
            }
            temp_normals.push_back(vn);
        }
        /* -------------------- FACE ---------------------------------------- */
        else if (command == "f")
        {
            std::vector<IndexTriplet> face_indices;      // v/vt/vn pentru fiecare vertex
            std::string face_chunk_str;                  // de ex.  "3/2/1"

            bool face_parse_error = false;

            /* ------- Citeşte fiecare "chunk" (v/vt/vn) din linia f -------- */
            while (ss >> face_chunk_str)
            {
                std::stringstream chunk_ss(face_chunk_str);
                std::string segment;
                IndexTriplet idx;      // se va popula cu v,vt,vn
                int part = 0;          // 0=v, 1=vt, 2=vn

                while (std::getline(chunk_ss, segment, '/'))
                {
                    if (!segment.empty())
                    {
                        try
                        {
                            int index = std::stoi(segment);

                            /* Acceptă indici negativi (OpenGL-style) */
                            if (index < 0)
                            {
                                if (part == 0)         index = static_cast<int>(temp_verts.size())      + index + 1;
                                else if (part == 1)    index = static_cast<int>(temp_texCoords.size())  + index + 1;
                                else if (part == 2)    index = static_cast<int>(temp_normals.size())    + index + 1;
                            }

                            if (part == 0)      idx.v  = index;
                            else if (part == 1) idx.vt = index;
                            else if (part == 2) idx.vn = index;
                        }
                        catch (const std::exception&)
                        {
                            std::cerr << "WARNING: Line " << line_number
                                      << ": Invalid index in face chunk '" << face_chunk_str << "'\n";
                            face_parse_error = true;
                            break;
                        }
                    }
                    ++part;
                    if (part > 2) break;   // ignoră orice altceva
                }

                if (face_parse_error) break;
                if (idx.v == 0)
                {
                    std::cerr << "WARNING: Line " << line_number
                              << ": Face vertex without position index.\n";
                    face_parse_error = true;
                    break;
                }

                face_indices.push_back(idx);
            } /* -------- terminat parsat chunk-uri ------------------------- */

            if (face_parse_error) { load_error = true; continue; }

            /* ------- Verifică să existe cel puţin 3 indici pentru un triunghi */
            if (face_indices.size() < 3)
            {
                std::cerr << "WARNING: Line " << line_number
                          << ": Face with less than 3 vertices.\n";
                load_error = true;
                continue;
            }

            /* ------- Triangulează poligonul (fan method) ------------------ */
            const IndexTriplet& idx0 = face_indices[0];

            for (size_t i = 1; i + 1 < face_indices.size(); ++i)
            {
                const IndexTriplet& idx1 = face_indices[i];
                const IndexTriplet& idx2 = face_indices[i + 1];

                /* ---- Validare indici poziţie ------------------------------ */
                auto validPos = [&](const IndexTriplet& id) -> bool
                {
                    return id.v  > 0 && id.v  <= static_cast<int>(temp_verts.size());
                };
                if (!validPos(idx0) || !validPos(idx1) || !validPos(idx2))
                {
                    std::cerr << "ERROR: Line " << line_number
                              << ": Vertex index out of range during triangulation.\n";
                    load_error = true;
                    break;
                }

                /* ---- Construieşte triunghiul ------------------------------ */
                triangle tri;

                tri.p[0] = temp_verts[idx0.v - 1];
                tri.p[1] = temp_verts[idx1.v - 1];
                tri.p[2] = temp_verts[idx2.v - 1];

                /* ---- UV: 0 dacă lipsesc ----------------------------------- */
                auto fetchUV = [&](int uvIdx) -> glm::vec2
                {
                    if (uvIdx > 0 && uvIdx <= static_cast<int>(temp_texCoords.size()))
                        return temp_texCoords[uvIdx - 1];
                    return glm::vec2(0.0f, 0.0f);
                };

                tri.t[0] = fetchUV(idx0.vt);
                tri.t[1] = fetchUV(idx1.vt);
                tri.t[2] = fetchUV(idx2.vt);

                tris.push_back(tri);           // <<< OUTPUT
            }
        } /* ---------------- END face ("f") -------------------------------- */
    } /* --------------------- END while file lines ------------------------- */

    f.close();

    /* --------- Mesaje finale de status ----------------------------------- */
    if (load_error)
        std::cerr << "NOTE: Warnings or errors occurred during OBJ load of '"
                  << sFilename << "'.\n";

    if (tris.empty())
        std::cerr << "WARNING: No triangles produced while loading '"
                  << sFilename << "'.\n";

    std::cout << "Finished loading '" << sFilename
              << "'. Triangles loaded: " << tris.size() << '\n';
    return !tris.empty();   // true doar dacă avem ceva de randat
}

// Input events
std::optional<InputEvent> Renderer2D::detectInputEvent()
{
    SDL_Event event;

    while (SDL_PollEvent(&event))
    {
        InputEvent input;

        switch (event.type)
        {
        case SDL_EVENT_KEY_DOWN:
            input.type = "KEYDOWN";
            input.key = event.key.key;
            break;
        case SDL_EVENT_KEY_UP:
            input.type = "KEYUP";
            input.key = event.key.key;
            break;
        case SDL_EVENT_MOUSE_MOTION:
            input.type = "MOUSEMOTION";
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            input.type = "MOUSEDOWN";
            input.key = event.button.button;
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            input.type = "MOUSEUP";
            input.key = event.button.button;
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            input.type = "MOUSEWHEEL";
            input.wheelY = event.wheel.y;
            input.wheelX = event.wheel.x;
            input.key = event.button.button;
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        default:
            return std::nullopt;
        }

        return input;
    }

    return std::nullopt;
    ;
}

std::vector<InputEvent> Renderer2D::poolInputEvents()
{
    std::vector<InputEvent> events;
    SDL_Event event;

    while (SDL_PollEvent(&event))
    {
        InputEvent input;

        switch (event.type)
        {
        case SDL_EVENT_QUIT:
            isRunning = false;
            return events;

        case SDL_EVENT_KEY_DOWN:
            input.type = "KEYDOWN";
            input.key = event.key.key;
            break;
        case SDL_EVENT_KEY_UP:
            input.type = "KEYUP";
            input.key = event.key.key;
            break;
        case SDL_EVENT_MOUSE_MOTION:
            input.type = "MOUSEMOTION";
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            input.type = "MOUSEDOWN";
            input.key = event.button.button;
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            input.type = "MOUSEUP";
            input.key = event.button.button;
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            input.type = "MOUSEWHEEL";
            input.wheelY = event.wheel.y;
            input.wheelX = event.wheel.x;
            input.key = event.button.button;
            input.mouseX = event.motion.x;
            input.mouseY = event.motion.y;
            break;
        default:
            continue;
        }

        events.push_back(input);
    }

    return events;
}
// TEXTURĂ: încarcă din disc + upload pe GPU                            
Texture* Renderer2D::loadTexture(const std::string& path)
{
    currentTex = new Texture(path, useGPU);     // ① încarcă în RAM și opțional în VRAM

    if (useGPU) {                               // ② trimite către kernel
        uploadTexture(currentTex->device, currentTex->w, currentTex->h);
        setTexturing(true);
    } else {
        setTexturing(false);
        std::cerr << "[WARN] CPU path încă nu e implementat pentru sampling!\n";
    }

    if (currentMesh)                            // ③ leagă de mesh curent
        currentMesh->texture = currentTex;

    return currentTex;                          // ownership rămâne în C++
}


void Renderer2D::setTexture(Texture* t)
{
    currentTex = t;
    if (useGPU) {
        uploadTexture(t->device, t->w, t->h);
        setTexturing(true);
    } else {
        setTexturing(false);
    }
}



static uint32_t sampleCPU(const Texture* tex, float u, float v)
{
    if (!tex) return 0xFFFFFFFFu;     // alb fallback
    u -= std::floor(u);               // repeat
    v -= std::floor(v);

    int x = int(u * tex->w);
    int y = int((1.f - v) * tex->h);
    x = std::clamp(x, 0, tex->w - 1);
    y = std::clamp(y, 0, tex->h - 1);

    return tex->pixels[y * tex->w + x];
}
void Renderer2D::fillTexturedTri(const triangle& tri, const Texture* tex)
{
    if (!tex) return;

    // --- 1) bounding-box clamped la ecran -------------------------
    int minX = int(std::floor(std::min({tri.p[0].x, tri.p[1].x, tri.p[2].x})));
    int maxX = int(std::ceil (std::max({tri.p[0].x, tri.p[1].x, tri.p[2].x})));
    int minY = int(std::floor(std::min({tri.p[0].y, tri.p[1].y, tri.p[2].y})));
    int maxY = int(std::ceil (std::max({tri.p[0].y, tri.p[1].y, tri.p[2].y})));

    minX = std::clamp(minX, 0, int(windowWidth )-1);
    maxX = std::clamp(maxX, 0, int(windowWidth )-1);
    minY = std::clamp(minY, 0, int(windowHeight)-1);
    maxY = std::clamp(maxY, 0, int(windowHeight)-1);

    // --- 2) edge-function (barycentric) pre-compute ---------------
    auto edge = [](const glm::vec4& a,const glm::vec4& b,float x,float y)
                { return (b.y-a.y)*(x-a.x) - (b.x-a.x)*(y-a.y); };
    const float denom = edge(tri.p[0], tri.p[1], tri.p[2].x, tri.p[2].y);
    if (denom == 0.0f) return;  // degenerat

    // pentru perspectivă-correct: 1/w + UV/w
    float w0Inv = 1.0f / tri.p[0].w;
    float w1Inv = 1.0f / tri.p[1].w;
    float w2Inv = 1.0f / tri.p[2].w;

    glm::vec2 uv0 = tri.t[0] * w0Inv;
    glm::vec2 uv1 = tri.t[1] * w1Inv;
    glm::vec2 uv2 = tri.t[2] * w2Inv;

    // --- 3) parcurgere pixeli -------------------------------------
    for (int y = minY; y <= maxY; ++y)
    for (int x = minX; x <= maxX; ++x)
    {
        float w0 = edge(tri.p[1], tri.p[2], x+0.5f, y+0.5f) / denom;
        float w1 = edge(tri.p[2], tri.p[0], x+0.5f, y+0.5f) / denom;
        float w2 = 1.0f - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;

        // adâncime interpolată
        float z = w0*tri.p[0].z + w1*tri.p[1].z + w2*tri.p[2].z;
        int   idx = y * windowWidth + x;
        if (z >= depthBufferCPU[idx]) continue;   // FAIL depth-test

        // --- 4) perspective-correct UV -----------------------------
        // --- 4) perspective-correct UV ----------------------------------
float invW = w0*w0Inv + w1*w1Inv + w2*w2Inv;
float u = (w0*uv0.x + w1*uv1.x + w2*uv2.x) / invW;
float v = (w0*uv0.y + w1*uv1.y + w2*uv2.y) / invW;

/*  ▲  adaugă aici 1 linie  */
u -= std::floor(u);         // repeat pe orizontală
v -= std::floor(v);         // repeat pe verticală

int tx = int(u * tex->w) & (tex->w - 1);           // repeat (funcţionează pt. texturi POT)
int ty = int((1.0f - v) * tex->h) & (tex->h - 1);  // idem






        uint32_t src = tex->pixels[ty * tex->w + tx];
        screenBuffer[idx]     = *reinterpret_cast<RGBA*>(&src);
        depthBufferCPU[idx]   = z;
    }
}
